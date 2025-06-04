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
from moviepy.editor import VideoFileClip
from models.raft.raft_wrapper import RAFTOpticalFlow

app = Flask(__name__, static_folder='static')
app.config['UPLOAD_FOLDER'] = os.path.join('static', 'uploads')
app.config['PROCESSED_FOLDER'] = os.path.join('static', 'processed')
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

# Upload klasörlerini oluştur
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['PROCESSED_FOLDER'], exist_ok=True)

zits_model = ZITSInpainter()
background_remover = BackgroundRemover()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def index():
    return render_template('home.html')

@app.route('/editor')
def editor():
    file_url = request.args.get('file_url', '')
    return render_template('editor.html', file_url=file_url)

@app.route('/remove-bg')
def remove_bg_page():
    return render_template('remove_bg.html')

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
            
            # Dosyayı kaydet
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
        # Dosya adını ve maske verilerini al
        filename = request.form.get('filename')
        mask_data = request.form.get('mask')
        
        if not filename or not mask_data:
            return jsonify({'success': False, 'error': 'Eksik veri'}), 400
        
        # Orijinal görüntüyü yükle
        img_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        if not os.path.exists(img_path):
            return jsonify({'success': False, 'error': 'Dosya bulunamadı'}), 404
        
        # Görüntüyü oku
        image = cv2.imread(img_path)
        if image is None:
            return jsonify({'success': False, 'error': 'Görüntü okunamadı'}), 400
        
        try:
            # Base64 maskesini numpy dizisine dönüştür
            mask_data = mask_data.split(',')[1]
            mask_bytes = base64.b64decode(mask_data)
            mask_arr = np.frombuffer(mask_bytes, np.uint8)
            mask = cv2.imdecode(mask_arr, cv2.IMREAD_GRAYSCALE)
            
            # Maskeyi görüntü boyutuna ayarla
            mask = cv2.resize(mask, (image.shape[1], image.shape[0]))
            
            # İnpainting işlemini gerçekleştir
            result = zits_model.inpaint(image, mask)
            
            # Sonucu kaydet
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
        
        # Orijinal görüntüyü yükle
        img_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        if not os.path.exists(img_path):
            return jsonify({'success': False, 'error': 'Dosya bulunamadı'}), 404
        
        # Görüntüyü oku
        image = cv2.imread(img_path)
        if image is None:
            return jsonify({'success': False, 'error': 'Görüntü okunamadı'}), 400
        
        try:
            # Arka planı kaldır
            result = background_remover.remove_background(image)
            
            # Eğer arka plan rengi saydam değilse, arka planı seçilen renkle doldur
            if bg_color != 'transparent':
                from PIL import ImageColor
                rgb_color = ImageColor.getrgb(bg_color)
                # RGBA'yı RGB'ye çevir, arka planı doldur
                pil_img = Image.fromarray(result)
                background = Image.new('RGBA', pil_img.size, rgb_color + (255,))
                background.paste(pil_img, mask=pil_img.split()[3])
                result = np.array(background.convert('RGB'))
                save_mode = 'JPEG'
                output_ext = 'jpg'
            else:
                # Saydam PNG olarak kaydet
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

@app.route('/draw-editor')
def draw_editor():
    file_type = request.args.get('type', 'image') # Default görsel editör
    return render_template('draw_editor.html', file_type=file_type)

@app.route('/draw_process_image', methods=['POST'])
def draw_process_image():
    try:
        file = request.files.get('file')
        draw_data = request.form.get('draw')
        if not file or not draw_data:
            return jsonify({'success': False, 'error': 'Eksik veri'}), 400
        # Orijinal fotoğrafı oku
        img = Image.open(file.stream).convert('RGBA')
        # Çizim maskesini base64'ten oku
        header, b64data = draw_data.split(',')
        mask_bytes = base64.b64decode(b64data)
        mask_img = Image.open(io.BytesIO(mask_bytes)).convert('RGBA')
        # Çizimi fotoğrafın üstüne uygula
        result = Image.alpha_composite(img, mask_img)
        # Sonucu kaydet
        output_filename = f"draw_{uuid.uuid4().hex}.png"
        output_path = os.path.join(app.config['PROCESSED_FOLDER'], output_filename)
        result.save(output_path, 'PNG')
        return jsonify({'success': True, 'result_url': url_for('static', filename=f'processed/{output_filename}')})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/draw_process_video', methods=['POST'])
def draw_process_video():
    try:
        file = request.files.get('file')
        draw_data = request.form.get('draw')
        if not file or not draw_data:
            return jsonify({'success': False, 'error': 'Eksik veri'}), 400
        # Video dosyasını kaydet
        video_filename = f"video_{uuid.uuid4().hex}.mp4"
        video_path = os.path.join(app.config['UPLOAD_FOLDER'], video_filename)
        file.save(video_path)
        # Çizim maskesini oku
        header, b64data = draw_data.split(',')
        mask_bytes = base64.b64decode(b64data)
        mask_img = Image.open(io.BytesIO(mask_bytes)).convert('RGBA')
        mask_np = np.array(mask_img)
        # RAFT modelini yükle
        raft_model_path = os.path.join('models', 'raft', 'raft-sintel.pth')
        raft = RAFTOpticalFlow(raft_model_path, device='cpu')
        # Videoyu karelere ayır
        cap = cv2.VideoCapture(video_path)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()
        # İlk kareye çizim maskesini uygula
        out_frames = []
        prev_mask = mask_np[..., :4]  # RGBA
        prev_frame = cv2.cvtColor(frames[0], cv2.COLOR_BGR2RGB)
        for i, frame in enumerate(frames):
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if i > 0:
                # Optik akış ile maskeyi bir sonraki kareye taşı
                flow = raft.calc_flow(prev_frame, rgb_frame)
                h, w = prev_mask.shape[:2]
                flow_map = np.stack(np.meshgrid(np.arange(w), np.arange(h)), axis=-1).astype(np.float32)
                flow_map += flow
                # Yeni maskeyi warp et
                new_mask = cv2.remap(prev_mask, flow_map[...,0], flow_map[...,1], interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
                prev_mask = new_mask
            # Maskeyi mevcut kareye uygula
            pil_frame = Image.fromarray(rgb_frame).convert('RGBA')
            pil_mask = Image.fromarray(prev_mask.astype(np.uint8), 'RGBA')
            merged = Image.alpha_composite(pil_frame, pil_mask)
            out_frames.append(np.array(merged.convert('RGB')))
            prev_frame = rgb_frame
        # Son videoyu kaydet
        output_filename = f"draw_{uuid.uuid4().hex}.mp4"
        output_path = os.path.join(app.config['PROCESSED_FOLDER'], output_filename)
        height, width, _ = out_frames[0].shape
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), 25, (width, height))
        for f in out_frames:
            out.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
        out.release()
        return jsonify({'success': True, 'result_url': url_for('static', filename=f'processed/{output_filename}')})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)