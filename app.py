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
import torch
import torchvision.transforms as transforms
from models.raft.raft import RAFT
import argparse
from datetime import datetime
import time

app = Flask(__name__, static_folder='static')
app.config['UPLOAD_FOLDER'] = os.path.join('static', 'uploads')
app.config['PROCESSED_FOLDER'] = os.path.join('static', 'processed')
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}
app.config['ALLOWED_VIDEO_EXTENSIONS'] = {'mp4', 'avi', 'mov', 'mkv'}

# Klasörleri oluştur
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

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def allowed_video_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_VIDEO_EXTENSIONS']

def load_raft_model():
    """RAFT Optical Flow modelini yükle"""
    try:
        print("RAFT modeli yükleniyor...")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"RAFT modeli için kullanılacak cihaz: {device}")
        
        args_dict = {
            'model': 'raft-things.pth',
            'small': False,
            'mixed_precision': True,
            'corr_levels': 4,
            'corr_radius': 4,
            'dropout': 0.0,
            'alternate_corr': False
        }
        
        args = argparse.Namespace(**args_dict)
        model_path = os.path.join('models/raft', args.model)
        
        if not os.path.exists(model_path):
            print(f"HATA: Model dosyası bulunamadı: {model_path}")
            return None
            
        model = RAFT(args)
        state_dict = torch.load(model_path, map_location=device)
        
        # DataParallel prefix temizle
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
        return None

# RAFT modelini başlat
try:
    raft_model = load_raft_model()
except Exception as e:
    print(f"RAFT model başlatma hatası: {e}")
    raft_model = None

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

def extract_drawing_regions(mask):
    """Çizim bölgelerini tespit et ve analiz et"""
    try:
        if len(mask.shape) == 3:
            gray_mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        else:
            gray_mask = mask
        
        # Binary threshold
        _, binary = cv2.threshold(gray_mask, 10, 255, cv2.THRESH_BINARY)
        
        # Morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        # Contourları bul
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        regions = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 50:  # Minimum alan
                x, y, w, h = cv2.boundingRect(contour)
                
                # Padding ekle
                padding = 10
                x = max(0, x - padding)
                y = max(0, y - padding)
                w = min(mask.shape[1] - x, w + 2 * padding)
                h = min(mask.shape[0] - y, h + 2 * padding)
                
                # Çizim mask'ını oluştur
                region_mask = np.zeros_like(gray_mask)
                cv2.drawContours(region_mask, [contour], -1, 255, -1)
                
                regions.append({
                    'bbox': (x, y, w, h),
                    'mask': region_mask,
                    'contour': contour,
                    'area': area,
                    'center': (x + w//2, y + h//2)
                })
        
        return regions
        
    except Exception as e:
        print(f"Çizim bölgesi tespit hatası: {e}")
        return []

def calculate_region_flow(flow, region):
    """Belirli bir bölge için dominant flow hesapla"""
    try:
        x, y, w, h = region['bbox']
        region_flow = flow[y:y+h, x:x+w]
        
        # Mask kullanarak sadece çizim alanındaki flow'u al
        mask = region['mask'][y:y+h, x:x+w] > 0
        
        if not np.any(mask):
            return np.array([0.0, 0.0])
        
        # Maskelenmiş flow vektörleri
        masked_flow_x = region_flow[mask, 0]
        masked_flow_y = region_flow[mask, 1]
        
        # Median flow (outlier'lara karşı robust)
        median_flow_x = np.median(masked_flow_x)
        median_flow_y = np.median(masked_flow_y)
        
        return np.array([median_flow_x, median_flow_y])
        
    except Exception as e:
        print(f"Bölge flow hesaplama hatası: {e}")
        return np.array([0.0, 0.0])

def apply_flow_to_region(region, flow_vector, frame_size):
    """Flow vektörünü kullanarak bölgeyi hareket ettir"""
    try:
        width, height = frame_size
        x, y, w, h = region['bbox']
        
        # Yeni pozisyon
        new_x = max(0, min(width - w, x + flow_vector[0]))
        new_y = max(0, min(height - h, y + flow_vector[1]))
        
        # Yeni bbox
        new_bbox = (int(new_x), int(new_y), w, h)
        
        # Yeni region oluştur
        new_region = region.copy()
        new_region['bbox'] = new_bbox
        new_region['center'] = (int(new_x + w//2), int(new_y + h//2))
        
        return new_region
        
    except Exception as e:
        print(f"Bölge hareket ettirme hatası: {e}")
        return region

def render_regions_to_frame(frame, regions, original_mask):
    """Bölgeleri frame üzerine çiz"""
    try:
        result = frame.copy().astype(np.float32)
        
        for region in regions:
            x, y, w, h = region['bbox']
            
            # Orijinal mask'tan bu bölgeyi al
            region_original_mask = region['mask']
            
            # Yeni pozisyona yerleştir
            if x + w <= frame.shape[1] and y + h <= frame.shape[0]:
                # Alpha blending
                alpha = 0.6
                
                # Mask'ı 3 kanala çevir
                if len(original_mask.shape) == 3:
                    mask_colored = cv2.resize(original_mask, (w, h))
                else:
                    mask_colored = cv2.resize(original_mask, (w, h))
                    mask_colored = cv2.cvtColor(mask_colored, cv2.COLOR_GRAY2BGR)
                
                # Binary mask oluştur
                region_resized = cv2.resize(region_original_mask, (w, h))
                binary_mask = (region_resized > 10).astype(np.float32) / 255.0
                binary_mask_3ch = np.stack([binary_mask] * 3, axis=2)
                
                # Blending uygula
                frame_region = result[y:y+h, x:x+w]
                mask_colored = mask_colored.astype(np.float32)
                
                blended = frame_region * (1 - binary_mask_3ch * alpha) + mask_colored * binary_mask_3ch * alpha
                result[y:y+h, x:x+w] = blended
        
        return np.clip(result, 0, 255).astype(np.uint8)
        
    except Exception as e:
        print(f"Region rendering hatası: {e}")
        return frame

def process_video_with_optical_flow_tracking(video_path, mask_image):
    """RAFT Optical Flow ile gelişmiş nesne takibi"""
    try:
        print("🎯 RAFT Optical Flow takipli video işleme başlatılıyor...")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise Exception("Video dosyası açılamadı")
        
        # Video bilgileri
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"📹 Video: {width}x{height}, {fps}fps, {total_frames} frame")
        
        # Çıktı dosyası - kaliteyi korumak için
        timestamp = str(int(time.time()))
        output_path = os.path.join('static/results', f'flow_tracked_{timestamp}.mp4')
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Yüksek kalite codec
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        if not out.isOpened():
            raise Exception("Video writer açılamadı")
        
        # İlk frame'i oku
        ret, first_frame = cap.read()
        if not ret:
            raise Exception("İlk frame okunamadı")
        
        # Mask'ı video boyutuna ölçekle
        mask_resized = cv2.resize(mask_image, (width, height))
        
        # Çizim bölgelerini tespit et
        regions = extract_drawing_regions(mask_resized)
        if not regions:
            print("❌ Çizim bölgesi bulunamadı, basit çizim uygulanacak")
            return apply_simple_overlay(video_path, mask_resized, output_path)
        
        print(f"✅ {len(regions)} çizim bölgesi tespit edildi")
        
        # İlk frame'e çizimi uygula
        first_frame_result = render_regions_to_frame(first_frame, regions, mask_resized)
        out.write(first_frame_result)
        
        # RAFT için device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Önceki frame
        prev_frame = first_frame.copy()
        current_regions = regions.copy()
        
        frame_count = 1
        
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
                if raft_model is not None:
                    prev_tensor = frame_to_tensor(prev_frame, device)
                    curr_tensor = frame_to_tensor(current_frame, device)
                    
                    if prev_tensor is not None and curr_tensor is not None:
                        with torch.no_grad():
                            _, flow_tensor = raft_model(prev_tensor, curr_tensor, iters=12, test_mode=True)
                        
                        flow = tensor_to_flow(flow_tensor)
                        
                        if flow is not None:
                            # Her bölge için flow hesapla ve güncelle
                            updated_regions = []
                            for region in current_regions:
                                region_flow = calculate_region_flow(flow, region)
                                updated_region = apply_flow_to_region(region, region_flow, (width, height))
                                updated_regions.append(updated_region)
                            
                            current_regions = updated_regions
                        else:
                                                    print(f"⚠️ Frame {frame_count}: Tensor çevrimi başarısız")
                else:
                    print(f"⚠️ Frame {frame_count}: RAFT modeli yok")
                
                # Güncellenmiş bölgelerle frame'i render et
                frame_result = render_regions_to_frame(current_frame, current_regions, mask_resized)
                out.write(frame_result)
                
                # Bir sonraki iterasyon için
                prev_frame = current_frame.copy()
                
            except Exception as frame_error:
                print(f"❌ Frame {frame_count} hatası: {frame_error}")
                # Hata durumunda önceki pozisyonları kullan
                frame_result = render_regions_to_frame(current_frame, current_regions, mask_resized)
                out.write(frame_result)
        
        cap.release()
        out.release()
        
        print(f"✅ Optical flow takipli video tamamlandı: {output_path}")
        return output_path
        
    except Exception as e:
        print(f"❌ Optical flow takip hatası: {str(e)}")
        print("Detay:", traceback.format_exc())
        return None

def apply_simple_overlay(video_path, mask, output_path):
    """Basit overlay (fallback)"""
    try:
        print("🔄 Basit overlay uygulanıyor...")
        
        cap = cv2.VideoCapture(video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_count += 1
            
            # Basit alpha blending
            if len(mask.shape) == 3:
                mask_colored = mask
            else:
                mask_colored = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            
            mask_binary = cv2.cvtColor(mask_colored, cv2.COLOR_BGR2GRAY)
            mask_binary = (mask_binary > 10).astype(np.float32) / 255.0
            mask_3ch = np.stack([mask_binary] * 3, axis=2)
            
            alpha = 0.6
            result = frame.astype(np.float32)
            mask_colored = mask_colored.astype(np.float32)
            
            result = result * (1 - mask_3ch * alpha) + mask_colored * mask_3ch * alpha
            
            out.write(np.clip(result, 0, 255).astype(np.uint8))
        
        cap.release()
        out.release()
        
        print(f"✅ Basit overlay tamamlandı: {output_path}")
        return output_path
        
    except Exception as e:
        print(f"❌ Basit overlay hatası: {e}")
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
        result_path = process_video_with_optical_flow_tracking(temp_video_path, mask_array)
        
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