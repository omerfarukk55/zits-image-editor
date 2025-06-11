import os
import base64
import io
import uuid
from flask import Flask, render_template, request, redirect, url_for, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from PIL import Image
import traceback
import cv2
import numpy as np
from io import BytesIO
from models.zits_model import ZITSInpainter
from models.background_remover import BackgroundRemover
from moviepy.editor import VideoFileClip, ImageSequenceClip
from models.raft.raft_wrapper import RAFTOpticalFlow
import torch
import torchvision.transforms as transforms  # Bu satırı ekleyin
from models.raft.raft import RAFT
import argparse
from datetime import datetime
import time
try:
    from scipy import ndimage
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("Scipy mevcut değil, OpenCV blur kullanılacak")

app = Flask(__name__, static_folder='static')
app.config['UPLOAD_FOLDER'] = os.path.join('static', 'uploads')
app.config['PROCESSED_FOLDER'] = os.path.join('static', 'processed')
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}
app.config['ALLOWED_VIDEO_EXTENSIONS'] = {'mp4', 'avi', 'mov', 'mkv'}

# Upload klasörlerini oluştur
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['PROCESSED_FOLDER'], exist_ok=True)
os.makedirs('temp', exist_ok=True)
os.makedirs('static/results', exist_ok=True)

# Modelleri başlat
try:
    print("ZITS modeli başlatılıyor...")
    zits_model = ZITSInpainter()
    print("ZITS modeli başarıyla başlatıldı")
except Exception as e:
    print(f"ZITS modeli başlatma hatası: {e}")
    zits_model = None

try:
    print("Arka plan kaldırma modeli başlatılıyor...")
    background_remover = BackgroundRemover()
    print("Arka plan kaldırma modeli başarıyla başlatıldı")
except Exception as e:
    print(f"Arka plan kaldırma modeli başlatma hatası: {e}")
    background_remover = None

# Global değişkenler
raft_model = None

def load_raft_model():
    try:
        print("RAFT modeli yükleniyor...")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"RAFT modeli için kullanılacak cihaz: {device}")
        
        # RAFT model yapılandırması
        args_dict = {
            'model': 'raft-things.pth',
            'small': False,
            'mixed_precision': True,
            'hidden_dims': [128]*3,
            'context_dims': [128]*3,
            'corr_implementation': 'reg',
            'corr_levels': 4,
            'corr_radius': 4,
            'mask_pred': False,
            'dropout': 0.0,
            'alternate_corr': False,
            'mask_pred_hidden_dims': [128]*3,
            'mask_pred_context_dims': [128]*3,
            'mask_pred_corr_levels': 4,
            'mask_pred_corr_radius': 4
        }
        
        args = argparse.Namespace(**args_dict)
        
        # Model dosyasının varlığını kontrol et
        model_path = os.path.join('models/raft', args.model)
        if not os.path.exists(model_path):
            print(f"HATA: Model dosyası bulunamadı: {model_path}")
            return None
            
        print(f"Model dosyası bulundu: {model_path}")
        
        model = RAFT(args)
        state_dict = torch.load(model_path, map_location=device)
        
        # DataParallel prefix'ini temizle
        if list(state_dict.keys())[0].startswith('module.'):
            new_state_dict = {}
            for key, value in state_dict.items():
                new_key = key.replace('module.', '')
                new_state_dict[new_key] = value
            state_dict = new_state_dict
        
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        
        print("RAFT modeli başarıyla yüklendi.")
        return model
        
    except Exception as e:
        print(f"RAFT modeli yüklenirken hata: {str(e)}")
        print("Hata detayı:", traceback.format_exc())
        return None

# RAFT modelini başlat
try:
    raft_model = load_raft_model()
except Exception as e:
    print(f"RAFT model başlatma hatası: {e}")
    raft_model = None

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def allowed_video_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_VIDEO_EXTENSIONS']

def frame_to_tensor(frame, device):
    """OpenCV frame'i RAFT için tensor'e çevir"""
    try:
        # BGR -> RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # PIL Image -> Tensor
        pil_image = Image.fromarray(frame_rgb)
        
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        
        tensor = transform(pil_image)
        tensor = tensor.unsqueeze(0)  # Batch dimension
        tensor = tensor.to(device)
        tensor = tensor * 255.0  # RAFT 0-255 bekliyor
        
        return tensor
        
    except Exception as e:
        print(f"Frame tensor çevirme hatası: {e}")
        return None

def tensor_to_flow(tensor):
    """RAFT tensor çıktısını flow array'e çevir"""
    try:
        if tensor.is_cuda:
            tensor = tensor.cpu()
        
        flow = tensor.detach().numpy()
        
        # [1, 2, H, W] -> [H, W, 2]
        if flow.ndim == 4 and flow.shape[0] == 1:
            flow = flow.squeeze(0)
        
        if flow.shape[0] == 2:
            flow = flow.transpose(1, 2, 0)
        
        return flow
        
    except Exception as e:
        print(f"Tensor flow çevirme hatası: {e}")
        return None

def apply_flow_to_mask(mask, flow):
    """Optik akışı kullanarak maskeyi hareket ettirir."""
    try:
        h, w = mask.shape[:2]
        y, x = np.mgrid[0:h, 0:w].astype(np.float32)
        
        flow_x = flow[..., 0]
        flow_y = flow[..., 1]
        new_x = x + flow_x
        new_y = y + flow_y
        
        new_x = np.clip(new_x, 0, w-1)
        new_y = np.clip(new_y, 0, h-1)
        
        new_mask = cv2.remap(mask, new_x, new_y, cv2.INTER_LINEAR)
        
        return new_mask
    except Exception as e:
        print(f"Maske hareket ettirme hatası: {str(e)}")
        return mask

def apply_mask_to_frame(frame, mask):
    """Mask'ı frame'e uygula"""
    try:
        # Mask'ın 3 kanallı olduğundan emin ol
        if len(mask.shape) == 2:
            mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        elif mask.shape[2] == 4:  # RGBA
            mask = cv2.cvtColor(mask, cv2.COLOR_RGBA2BGR)
        
        # Frame boyutuna uygun hale getir
        if mask.shape[:2] != frame.shape[:2]:
            mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
        
        # Alpha blending
        alpha = 0.7
        
        # Mask'ın siyah olmayan kısımlarını bul
        mask_binary = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        mask_binary = (mask_binary > 10).astype(np.uint8) * 255
        
        # 3 kanala çevir
        mask_3ch = cv2.merge([mask_binary, mask_binary, mask_binary]) / 255.0
        
        # Blending uygula
        result = frame.astype(np.float32)
        mask_colored = mask.astype(np.float32)
        
        result = result * (1 - mask_3ch * alpha) + mask_colored * mask_3ch * alpha
        
        return result.astype(np.uint8)
        
    except Exception as e:
        print(f"Mask uygulama hatası: {e}")
        return frame

def apply_flow_to_mask_improved(mask, flow, max_flow_magnitude=30, smooth_factor=0.7):
    """Düzeltilmiş gelişmiş mask hareket ettirme"""
    try:
        h, w = mask.shape[:2]
        
        # 1. Flow magnitude kontrolü
        flow_magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
        valid_flow_mask = flow_magnitude < max_flow_magnitude
        
        # 2. Outlier'ları temizle
        flow_filtered = flow.copy()
        flow_filtered[~valid_flow_mask] = 0
        
        # 3. Flow'u smooth et - Scipy kontrolü
        if SCIPY_AVAILABLE:
            try:
                flow_filtered[..., 0] = ndimage.gaussian_filter(flow_filtered[..., 0], sigma=1.0)
                flow_filtered[..., 1] = ndimage.gaussian_filter(flow_filtered[..., 1], sigma=1.0)
            except:
                # Scipy hatası durumunda OpenCV kullan
                flow_filtered[..., 0] = cv2.GaussianBlur(flow_filtered[..., 0], (5, 5), 1.0)
                flow_filtered[..., 1] = cv2.GaussianBlur(flow_filtered[..., 1], (5, 5), 1.0)
        else:
            # Scipy yoksa OpenCV blur kullan
            flow_filtered[..., 0] = cv2.GaussianBlur(flow_filtered[..., 0], (5, 5), 1.0)
            flow_filtered[..., 1] = cv2.GaussianBlur(flow_filtered[..., 1], (5, 5), 1.0)
        
        # 4. Mask'ı grayscale'e çevir
        if len(mask.shape) == 3:
            mask_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        else:
            mask_gray = mask
        
        # 5. Aktif bölgeleri bul
        active_mask = mask_gray > 10
        
        if not np.any(active_mask):
            return mask
        
        # 6. Aktif bölgedeki flow'u hesapla
        active_indices = np.where(active_mask)
        if len(active_indices[0]) == 0:
            return mask
        
        active_flow_x = flow_filtered[active_indices[0], active_indices[1], 0]
        active_flow_y = flow_filtered[active_indices[0], active_indices[1], 1]
        
        # 7. Median flow hesapla (robust)
        median_flow_x = np.median(active_flow_x)
        median_flow_y = np.median(active_flow_y)
        
        # 8. Flow'u sınırla ve yumuşat
        median_flow_x = np.clip(median_flow_x, -max_flow_magnitude, max_flow_magnitude) * smooth_factor
        median_flow_y = np.clip(median_flow_y, -max_flow_magnitude, max_flow_magnitude) * smooth_factor
        
        # 9. Koordinat gridleri
        y_coords, x_coords = np.mgrid[0:h, 0:w].astype(np.float32)
        
        # 10. Yeni koordinatlar
        new_x = x_coords + median_flow_x
        new_y = y_coords + median_flow_y
        
        # 11. Sınırları kontrol et
        new_x = np.clip(new_x, 0, w-1)
        new_y = np.clip(new_y, 0, h-1)
        
        # 12. Mask'ı hareket ettir
        new_mask = cv2.remap(mask, new_x, new_y, cv2.INTER_LINEAR)
        
        return new_mask
        
    except Exception as e:
        print(f"Gelişmiş mask hareket ettirme hatası: {e}")
        return mask
def calculate_temporal_consistency_fixed(previous_masks, weight_decay=0.8):
    """Düzeltilmiş temporal tutarlılık"""
    try:
        if len(previous_masks) == 0:
            return None
        
        # İlk mask'ın boyutlarını referans al
        reference_mask = previous_masks[-1]
        
        # Referans mask'ı grayscale'e çevir ve boyutları al
        if len(reference_mask.shape) == 3:
            reference_gray = cv2.cvtColor(reference_mask, cv2.COLOR_BGR2GRAY)
            h, w = reference_gray.shape
        else:
            reference_gray = reference_mask
            h, w = reference_mask.shape
        
        # Birleştirme için grayscale array oluştur
        combined_mask = np.zeros((h, w), dtype=np.float32)
        total_weight = 0
        
        # Son 5 mask'ı işle
        for i, mask in enumerate(reversed(previous_masks[-5:])):
            weight = weight_decay ** i
            
            # Mask'ı grayscale'e çevir
            if len(mask.shape) == 3:
                mask_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            else:
                mask_gray = mask.copy()
            
            # Boyut uyumluluğunu kontrol et ve düzelt
            if mask_gray.shape != (h, w):
                mask_gray = cv2.resize(mask_gray, (w, h))
            
            # Ağırlıklı toplama
            combined_mask += mask_gray.astype(np.float32) * weight
            total_weight += weight
        
        # Normaliz et
        if total_weight > 0:
            combined_mask /= total_weight
        
        # Threshold uygula ve uint8'e çevir
        combined_mask = (combined_mask > 50).astype(np.uint8) * 255
        
        return combined_mask
        
    except Exception as e:
        print(f"Temporal consistency hatası: {e}")
        return None

def process_video_with_improved_tracking(video_path, mask_image):
    """Düzeltilmiş gelişmiş optical flow takibi"""
    try:
        print("🎯 Düzeltilmiş RAFT Optical Flow takibi başlatılıyor...")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise Exception("Video dosyası açılamadı")
        
        # Video bilgileri
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"📹 Video: {width}x{height}, {fps}fps, {total_frames} frame")
        
        # Çıktı dosyası
        timestamp = str(int(time.time()))
        output_path = os.path.join('static/results', f'fixed_tracking_{timestamp}.mp4')
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        if not out.isOpened():
            raise Exception("Video writer açılamadı")
        
        # Mask'ı hazırla - boyut standartlaştırması
        print(f"🎨 Orijinal mask shape: {mask_image.shape}")
        
        if len(mask_image.shape) == 3 and mask_image.shape[2] == 4:  # RGBA
            mask_resized = cv2.resize(mask_image, (width, height))
            mask_resized = cv2.cvtColor(mask_resized, cv2.COLOR_RGBA2BGR)
        elif len(mask_image.shape) == 3:  # RGB/BGR
            mask_resized = cv2.resize(mask_image, (width, height))
        else:  # Grayscale
            mask_resized = cv2.resize(mask_image, (width, height))
            mask_resized = cv2.cvtColor(mask_resized, cv2.COLOR_GRAY2BGR)
        
        print(f"🎨 Yeniden boyutlandırılmış mask: {mask_resized.shape}")
        
        # İlk frame'i oku
        ret, prev_frame = cap.read()
        if not ret:
            raise Exception("İlk frame okunamadı")
        
        # Değişkenler
        current_mask = mask_resized.copy()
        mask_history = []
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # İlk frame'i işle
        first_frame_result = apply_mask_to_frame(prev_frame, current_mask)
        out.write(first_frame_result)
        mask_history.append(current_mask.copy())
        
        frame_count = 1
        
        # RAFT modeli kontrolü
        if raft_model is None:
            print("⚠️ RAFT modeli yok, basit işleme uygulanıyor...")
            return process_video_simple_drawing(cap, out, mask_resized, total_frames, output_path)
        
        while True:
            ret, current_frame = cap.read()
            if not ret:
                break
                
            frame_count += 1
            
            # Progress
            if frame_count % 30 == 0:
                print(f"📊 İşleniyor: {frame_count}/{total_frames}")
            
            try:
                # RAFT ile optical flow hesapla
                prev_tensor = frame_to_tensor(prev_frame, device)
                curr_tensor = frame_to_tensor(current_frame, device)
                
                if prev_tensor is not None and curr_tensor is not None:
                    with torch.no_grad():
                        _, flow_tensor = raft_model(prev_tensor, curr_tensor, iters=12, test_mode=True)
                    
                    flow = tensor_to_flow(flow_tensor)
                    
                    if flow is not None:
                        # Flow'u mask boyutuna ölçekle
                        if flow.shape[:2] != current_mask.shape[:2]:
                            flow_resized = cv2.resize(flow, (current_mask.shape[1], current_mask.shape[0]))
                        else:
                            flow_resized = flow
                        
                        # Düzeltilmiş mask hareket ettirme
                        current_mask = apply_flow_to_mask_improved(
                            current_mask, 
                            flow_resized, 
                            max_flow_magnitude=25,  # Daha konservatif
                            smooth_factor=0.8       # Daha smooth
                        )
                        
                        # Temporal tutarlılık (düzeltilmiş)
                        if len(mask_history) > 2:
                            temporal_mask = calculate_temporal_consistency_fixed(mask_history[-3:])
                            if temporal_mask is not None:
                                # Mevcut mask'ı grayscale'e çevir
                                if len(current_mask.shape) == 3:
                                    current_mask_gray = cv2.cvtColor(current_mask, cv2.COLOR_BGR2GRAY)
                                else:
                                    current_mask_gray = current_mask.copy()
                                
                                # Boyut kontrolü
                                if temporal_mask.shape != current_mask_gray.shape:
                                    temporal_mask = cv2.resize(temporal_mask, 
                                                             (current_mask_gray.shape[1], current_mask_gray.shape[0]))
                                
                                # Ağırlıklı birleştirme
                                alpha = 0.6
                                combined = (current_mask_gray.astype(np.float32) * alpha + 
                                          temporal_mask.astype(np.float32) * (1-alpha)).astype(np.uint8)
                                
                                # Geri BGR'e çevir
                                if len(current_mask.shape) == 3:
                                    current_mask = cv2.cvtColor(combined, cv2.COLOR_GRAY2BGR)
                                else:
                                    current_mask = combined
                        
                        # Mask geçmişini güncelle
                        mask_history.append(current_mask.copy())
                        if len(mask_history) > 5:  # Sadece son 5 mask
                            mask_history.pop(0)
                    
                    else:
                        print(f"⚠️ Frame {frame_count}: Flow hesaplanamadı")
                else:
                    print(f"⚠️ Frame {frame_count}: Tensor çevrimi başarısız")
                
                # Frame'i işle
                frame_result = apply_mask_to_frame(current_frame, current_mask)
                out.write(frame_result)
                
                # Bir sonraki iterasyon için
                prev_frame = current_frame.copy()
                
            except Exception as frame_error:
                print(f"❌ Frame {frame_count} hatası: {frame_error}")
                # Hata durumunda önceki mask'ı kullan
                frame_result = apply_mask_to_frame(current_frame, current_mask)
                out.write(frame_result)
        
        cap.release()
        out.release()
        
        print(f"✅ Düzeltilmiş takip tamamlandı: {output_path}")
        return output_path
        
    except Exception as e:
        print(f"❌ Düzeltilmiş takip hatası: {str(e)}")
        print("Detay:", traceback.format_exc())
        return None

def process_video_simple_drawing(cap, out, mask, total_frames, output_path):
    """RAFT olmadan basit çizim uygulama"""
    try:
        print("Basit çizim uygulanıyor...")
        
        frame_count = 1
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_count += 1
            
            if frame_count % 50 == 0:
                print(f"Basit frame işleniyor: {frame_count}/{total_frames}")
            
            frame_with_drawing = apply_mask_to_frame(frame, mask)
            out.write(frame_with_drawing)
        
        cap.release()
        out.release()
        
        print(f"Basit video işleme tamamlandı: {output_path}")
        return output_path
        
    except Exception as e:
        print(f"Basit çizim hatası: {e}")
        if cap:
            cap.release()
        if out:
            out.release()
        return None

# Flask Routes
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/editor')
def editor():
    try:
        file_url = request.args.get('file_url', '')
        return render_template('editor.html', file_url=file_url)
    except Exception as e:
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/remove-bg')
def remove_bg_page():
    try:
        return render_template('remove_bg.html')
    except Exception as e:
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'Dosya yüklenmedi'})
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'success': False, 'error': 'Dosya seçilmedi'})
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            
            file.save(file_path)
            
            return jsonify({
                'success': True,
                'filename': filename,
                'file_url': url_for('static', filename=f"uploads/{filename}")
            })
        
        return jsonify({'success': False, 'error': 'İzin verilmeyen dosya türü'})
    except Exception as e:
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)})

@app.route('/download/<filename>')
def download_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename, as_attachment=True)

@app.route('/process_image', methods=['POST'])
def process_image():
    try:
        filename = request.form.get('filename')
        mask_data = request.form.get('mask')
        
        if not filename or not mask_data:
            return jsonify({'success': False, 'error': 'Eksik veri'}), 400
        
        img_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        if not os.path.exists(img_path):
            return jsonify({'success': False, 'error': 'Dosya bulunamadı'}), 404
        
        image = cv2.imread(img_path)
        if image is None:
            return jsonify({'success': False, 'error': 'Görüntü okunamadı'}), 400
        
        try:
            mask_data = mask_data.split(',')[1]
            mask_bytes = base64.b64decode(mask_data)
            mask_arr = np.frombuffer(mask_bytes, np.uint8)
            mask = cv2.imdecode(mask_arr, cv2.IMREAD_GRAYSCALE)
            
            mask = cv2.resize(mask, (image.shape[1], image.shape[0]))
            
            result = zits_model.inpaint(image, mask)
            
            output_filename = f"processed_{uuid.uuid4().hex}.png"
            output_path = os.path.join(app.config['PROCESSED_FOLDER'], output_filename)
            cv2.imwrite(output_path, result)
            
            return jsonify({
                'success': True,
                'processed_image': url_for('static', filename=f'processed/{output_filename}')
            })
            
        except Exception as e:
            print(f"Hata: {str(e)}")
            return jsonify({'success': False, 'error': str(e)}), 500
            
    except Exception as e:
        print(f"Genel hata: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/remove_background', methods=['POST'])
def remove_background():
    try:
        filename = request.form.get('filename')
        bg_color = request.form.get('bg_color', 'transparent')
        
        if not filename:
            return jsonify({'success': False, 'error': 'Dosya adı belirtilmedi'}), 400
        
        img_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        if not os.path.exists(img_path):
            return jsonify({'success': False, 'error': 'Dosya bulunamadı'}), 404
        
        image = cv2.imread(img_path)
        if image is None:
            return jsonify({'success': False, 'error': 'Görüntü okunamadı'}), 400
        
        try:
            result = background_remover.remove_background(image)
            
            if bg_color != 'transparent':
                from PIL import ImageColor
                rgb_color = ImageColor.getrgb(bg_color)
                pil_img = Image.fromarray(result)
                background = Image.new('RGBA', pil_img.size, rgb_color + (255,))
                background.paste(pil_img, mask=pil_img.split()[3])
                result = np.array(background.convert('RGB'))
                save_mode = 'JPEG'
                output_ext = 'jpg'
            else:
                save_mode = 'PNG'
                output_ext = 'png'
            
            output_filename = f"nobg_{uuid.uuid4().hex}.{output_ext}"
            output_path = os.path.join(app.config['PROCESSED_FOLDER'], output_filename)
            Image.fromarray(result).save(output_path, save_mode)
            
            return jsonify({
                'success': True,
                'processed_image': url_for('static', filename=f'processed/{output_filename}')
            })
            
        except Exception as e:
            print(f"Hata: {str(e)}")
            return jsonify({'success': False, 'error': str(e)}), 500
            
    except Exception as e:
        print(f"Genel hata: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/img-draw-editor')
def img_draw_editor():
    return render_template('imgdraw_editor.html')

@app.route('/vid-draw-editor')
def vid_draw_editor():
    return render_template('viddraw_editor.html')

@app.route('/draw_process_image', methods=['POST'])
def draw_process_image():
    try:
        if 'original_file' not in request.files or 'draw_data_url' not in request.form:
            return jsonify({'success': False, 'error': 'Dosya veya çizim verisi eksik'}), 400
        
        original_file = request.files['original_file']
        draw_data_url = request.form['draw_data_url']
        
        if original_file.filename == '':
            return jsonify({'success': False, 'error': 'Dosya seçilmedi'}), 400
        
        if '.' not in original_file.filename or original_file.filename.rsplit('.', 1)[1].lower() not in app.config['ALLOWED_EXTENSIONS']:
            return jsonify({'success': False, 'error': 'Desteklenmeyen dosya formatı'}), 400
        
        filename = secure_filename(original_file.filename)
        unique_filename = f"{uuid.uuid4().hex}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        original_file.save(filepath)
        
        try:
            draw_data_header, draw_data_base64 = draw_data_url.split(',', 1)
            draw_bytes = base64.b64decode(draw_data_base64)
            draw_img = Image.open(io.BytesIO(draw_bytes)).convert('RGBA')
        except Exception as e:
            if os.path.exists(filepath):
                os.remove(filepath)
            return jsonify({'success': False, 'error': f'Çizim verisi işlenirken hata: {str(e)}'}), 400

        try:
            img = Image.open(filepath).convert('RGBA')
        except Exception as e:
            if os.path.exists(filepath):
                os.remove(filepath)
            return jsonify({'success': False, 'error': f'Orijinal görsel açılırken hata: {str(e)}'}), 400
        
        if draw_img.size != img.size:
            draw_img = draw_img.resize(img.size, Image.Resampling.LANCZOS)
        result = Image.alpha_composite(img, draw_img)
        
        result_filename = 'processed_' + unique_filename
        result_path = os.path.join(app.config['PROCESSED_FOLDER'], result_filename)
        
        result.save(result_path, 'PNG')
        
        if os.path.exists(filepath):
            os.remove(filepath)
        
        return jsonify({
            'success': True,
            'result_url': url_for('static', filename=f'processed/{result_filename}')
        })
        
    except Exception as e:
        traceback.print_exc()
        return jsonify({'success': False, 'error': f'Sunucu hatası: {str(e)}'}), 500

@app.route('/draw_process_video', methods=['POST'])
def draw_process_video():
    """Gelişmiş nesne takipli video işleme"""
    try:
        print("🎬 Video işleme isteği alındı")
        
        video_file = request.files.get('video_file')
        draw_mask_url = request.form.get('draw_mask_url')
        
        if not video_file or not draw_mask_url:
            return jsonify({'success': False, 'error': 'Video dosyası ve çizim maskesi gerekli'})
        
        # Geçici dosya oluştur
        temp_video_path = os.path.join('temp', secure_filename(video_file.filename))
        os.makedirs('temp', exist_ok=True)
        video_file.save(temp_video_path)
        print(f"📁 Video geçici olarak kaydedildi: {temp_video_path}")
        
        # Mask'ı decode et
        try:
            mask_data = draw_mask_url.split(',')[1]
            mask_bytes = base64.b64decode(mask_data)
            mask_image = Image.open(io.BytesIO(mask_bytes))
            mask_array = np.array(mask_image)
            print(f"🎨 Mask işlendi: {mask_array.shape}")
        except Exception as mask_error:
            return jsonify({'success': False, 'error': f'Mask işleme hatası: {str(mask_error)}'})
        
        # Gelişmiş optical flow takipli video işle
        result_path = process_video_with_improved_tracking(temp_video_path, mask_array)
        # Geçici dosyayı temizle
        try:
            os.remove(temp_video_path)
            print("🗑️ Geçici dosya silindi")
        except:
            pass
        
        if result_path and os.path.exists(result_path):
            print(f"✅ Video işleme başarıyla tamamlandı: {result_path}")
            return jsonify({
                'success': True, 
                'result_url': '/' + result_path.replace('\\', '/')
            })
        else:
            print("❌ Video işleme başarısız")
            return jsonify({'success': False, 'error': 'Video işleme başarısız'})
            
    except Exception as e:
        print(f"❌ Video işlenirken hata oluştu: {str(e)}")
        print("Detay:", traceback.format_exc())
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)