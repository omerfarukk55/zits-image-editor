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

app = Flask(__name__, static_folder='static')
app.config['UPLOAD_FOLDER'] = os.path.join('static', 'uploads')
app.config['PROCESSED_FOLDER'] = os.path.join('static', 'processed')
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

# Upload klasörlerini oluştur
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['PROCESSED_FOLDER'], exist_ok=True)

zits_model = ZITSInpainter()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def index():
    return render_template('editor.html')

@app.route('/editor')
def editor():
    file_url = request.args.get('file_url', '')
    return render_template('editor.html', file_url=file_url)

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

if __name__ == '__main__':
    app.run(debug=True)