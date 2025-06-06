let image = null;
let isDrawing = false;
let drawMode = 'draw';
let brushColor = '#ff0000';
let brushSize = 5;
let lastX = 0, lastY = 0;
let textToAdd = '';
let brushType = 'normal';
let shapeMode = null; // 'rect' veya 'circle'
let startShape = null;
let isShift = false;
let filename = null; // Yüklenen dosyanın adını saklamak için

// Çizim geçmişi için dizi (global tanımla)
let drawHistory = [];

// Fonksiyon: Çizim durumunu geçmişe kaydet
function saveDrawState() {
    // Mevcut canvas durumunu kaydet
    drawHistory.push(drawCtx.getImageData(0, 0, drawCanvas.width, drawCanvas.height));
    // Çok fazla geçmiş kaydetmemek için limitle
    if (drawHistory.length > 20) {
        drawHistory.shift();
    }
}

const fileInput = document.getElementById('fileInput');
const mainCanvas = document.getElementById('mainCanvas');
const drawCanvas = document.getElementById('drawCanvas');
const canvasWrap = document.getElementById('canvas-container'); // canvasWrap yerine canvas-container kullan
const ctx = mainCanvas.getContext('2d'); // mainCtx yerine ctx kullanıyorum tutarlılık için
const drawCtx = drawCanvas.getContext('2d');
const imgCtx = mainCanvas.getContext('2d'); // mainCanvas arka plan görseli için kullanılacak

// Elementler
const uploadArea = document.getElementById('upload-area');
const canvasContainer = document.getElementById('canvas-container');
const toolbar = document.querySelector('.toolbar');
const resultContainer = document.getElementById('resultContainer');
const resultImage = document.getElementById('resultImage');
const imageDownloadBtn = document.getElementById('imageDownloadBtn');
const loadingSpinner = document.getElementById('loading'); // base.html'deki loading spinner

// Araç Çubuğu Butonları ve Kontroller
const drawBtn = document.getElementById('drawBtn');
const textBtn = document.getElementById('textBtn');
const textInputWrap = document.getElementById('textInputWrap');
const textInput = document.getElementById('textInput');
const colorPicker = document.getElementById('colorPicker');
const brushSizeControl = document.getElementById('brushSize');
const brushSizeValueSpan = document.getElementById('brushSizeValue');
const brushTypeSelect = document.getElementById('brushType');
const rectBtn = document.getElementById('rectBtn');
const circleBtn = document.getElementById('circleBtn');
const clearBtn = document.getElementById('clearBtn');
const resetBtn = document.getElementById('resetBtn');
const undoBtn = document.getElementById('undoBtn');
const saveBtn = document.getElementById('saveBtn');
const newEditBtn = document.getElementById('newEditBtn');

// Olay Dinleyiciler
fileInput.addEventListener('change', handleFileSelect);
drawCanvas.addEventListener('mousedown', handleMouseDown);
drawCanvas.addEventListener('mousemove', handleMouseMove);
drawCanvas.addEventListener('mouseup', handleMouseUp);
drawCanvas.addEventListener('mouseout', handleMouseOut);
drawCanvas.addEventListener('click', handleCanvasClick);

drawBtn.addEventListener('click', () => setDrawMode('draw'));
textBtn.addEventListener('click', () => setDrawMode('text'));
colorPicker.addEventListener('input', (e) => brushColor = e.target.value);
brushSizeControl.addEventListener('input', handleBrushSizeChange);
brushTypeSelect.addEventListener('change', (e) => brushType = e.target.value);
rectBtn.addEventListener('click', () => setShapeMode('rect'));
circleBtn.addEventListener('click', () => setShapeMode('circle'));
clearBtn.addEventListener('click', clearDrawCanvas);
resetBtn.addEventListener('click', resetCanvas);
undoBtn.addEventListener('click', undoLastDraw);
saveBtn.addEventListener('click', handleSaveClick);
newEditBtn.addEventListener('click', resetPage);

document.addEventListener('keydown', (e) => { if(e.key === 'Shift') isShift = true; });
document.addEventListener('keyup', (e) => { if(e.key === 'Shift') isShift = false; });

// Dosya Seçimi İşleyici
function handleFileSelect(e) {
    const file = e.target.files[0];
    if (!file) {
        resetPage(); // Dosya seçimi iptal edilirse sayfayı sıfırla
        return;
    }

    if (!file.type.startsWith('image/')) {
        alert("Lütfen bir görsel dosyası seçin.");
        fileInput.value = ''; // Dosya seçimini temizle
        resetPage();
        return;
    }

    showLoading('Görsel yükleniyor...');

    const reader = new FileReader();
    reader.onload = function(event) {
        const img = new Image();
        img.onload = function() {
            hideLoading();

            const maxWidth = 800; // Maksimum genişlik
            let width = img.width;
            let height = img.height;

            if (width > maxWidth) {
                const ratio = maxWidth / width;
                width = maxWidth;
                height = height * ratio;
            }

            // Canvas boyutlarını ayarla
            mainCanvas.width = width;
            mainCanvas.height = height;
            drawCanvas.width = width;
            drawCanvas.height = height;

            // Canvas container'ın boyutlarını ayarla
            canvasContainer.style.width = width + 'px';
            canvasContainer.style.height = height + 'px';

            // Görseli mainCanvas'a çiz
            imgCtx.clearRect(0, 0, mainCanvas.width, mainCanvas.height);
            imgCtx.drawImage(img, 0, 0, width, height);

            // Çizim canvas'ını temizle
            drawCtx.clearRect(0, 0, drawCanvas.width, drawCanvas.height);
            drawHistory = []; // Çizim geçmişini sıfırla
            image = img; // Yüklenen görseli sakla (sıfırlama için)
            filename = file.name; // Dosya adını sakla

            // Alanları göster/gizle
            uploadArea.style.display = 'none';
            canvasContainer.style.display = 'block';
            toolbar.style.display = 'flex'; // Araç çubuğunu göster
             setDrawMode('draw'); // Varsayılan mod çizim

        };
        img.onerror = function() {
            console.error("Görsel yüklenirken bir hata oluştu.");
            alert("Görsel yüklenirken bir hata oluştu.");
            hideLoading();
            resetPage();
        };
        img.src = event.target.result; // Base64 data URL kullan
    };
     reader.onerror = function() {
         console.error("Dosya okuma hatası.");
         alert("Dosya okunurken bir hata oluştu.");
         hideLoading();
         resetPage();
     };

    reader.readAsDataURL(file); // Dosyayı Base64 olarak oku
}

// Çizim Modu Ayarlama
function setDrawMode(mode) {
    drawMode = mode;
    if (drawMode === 'draw') {
        drawBtn.classList.add('active');
        textBtn.classList.remove('active');
        textInputWrap.style.display = 'none';
        drawCanvas.style.cursor = 'crosshair';
    } else if (drawMode === 'text') {
        textBtn.classList.add('active');
        drawBtn.classList.remove('active');
        textInputWrap.style.display = 'flex';
        drawCanvas.style.cursor = 'text';
    }
}

// Fırça Boyutu Değişikliği İşleyici
function handleBrushSizeChange(e) {
    brushSize = parseInt(e.target.value);
    brushSizeValueSpan.textContent = brushSize + 'px';
}

// Şekil Modu Ayarlama
function setShapeMode(mode) {
    shapeMode = mode;
    // Şekil butonlarının aktif/pasif durumunu ayarlayın
    // Örneğin: rectBtn.classList.add('active'); ...
}

// Çizim Canvas'ını Temizleme
function clearDrawCanvas() {
    drawCtx.clearRect(0, 0, drawCanvas.width, drawCanvas.height);
    drawHistory = [];
}

// Canvas'ı Sıfırlama (Arka plan görselini yeniden çiz ve çizimleri temizle)
function resetCanvas() {
    if (image) {
        // Orijinal görseli yeniden çiz
         imgCtx.clearRect(0, 0, mainCanvas.width, mainCanvas.height);
         imgCtx.drawImage(image, 0, 0, mainCanvas.width, mainCanvas.height);

        drawCtx.clearRect(0, 0, drawCanvas.width, drawCanvas.height);
        drawHistory = [];
    } else {
        alert("Lütfen önce bir görsel yükleyin.");
    }
}

// Son Çizimi Geri Alma
function undoLastDraw() {
    if (drawHistory.length > 0) {
        // Son kaydedilen durumu geri yükle
        const lastState = drawHistory.pop();
        drawCtx.clearRect(0, 0, drawCanvas.width, drawCanvas.height); // Canvas'ı temizle
        drawCtx.putImageData(lastState, 0, 0); // Son durumu çiz
    }
}

// Çizim Olay İşleyicileri
function handleMouseDown(e) {
    if (drawMode === 'draw' && shapeMode === null) {
        isDrawing = true;
        saveDrawState(); // Çizime başlamadan önceki durumu kaydet
        const rect = drawCanvas.getBoundingClientRect();
        [lastX, lastY] = [e.clientX - rect.left, e.clientY - rect.top];
    } else if (drawMode === 'draw' && shapeMode !== null) {
         isDrawing = true;
         saveDrawState(); // Şekil çizmeye başlamadan önceki durumu kaydet
         const rect = drawCanvas.getBoundingClientRect();
         startShape = { x: e.clientX - rect.left, y: e.clientY - rect.top };
    }
}

function handleMouseMove(e) {
    if (!isDrawing) return;
    const rect = drawCanvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    if (drawMode === 'draw' && shapeMode === null) { // Serbest çizim
        drawCtx.strokeStyle = brushColor;
        drawCtx.lineWidth = brushSize;
        drawCtx.lineCap = 'round';
        drawCtx.lineJoin = 'round';
        drawCtx.shadowBlur = (brushType === 'glow') ? 10 : 0;
        drawCtx.shadowColor = (brushType === 'glow') ? brushColor : 'transparent';

        if (isShift) { // Düz çizgi çizme
            drawCtx.beginPath();
            drawCtx.moveTo(lastX, lastY);
            drawCtx.lineTo(x, y);
            drawCtx.stroke();
        } else if (brushType === 'dotted') { // Kesikli çizim
            drawCtx.beginPath();
            drawCtx.setLineDash([5, 10]);
            drawCtx.moveTo(lastX, lastY);
            drawCtx.lineTo(x, y);
            drawCtx.stroke();
            drawCtx.setLineDash([]); // Çizgi stili sıfırla
            [lastX, lastY] = [x, y];
        } else { // Normal çizim
            drawCtx.beginPath();
            drawCtx.moveTo(lastX, lastY);
            drawCtx.lineTo(x, y);
            drawCtx.stroke();
            [lastX, lastY] = [x, y];
        }

    } else if (drawMode === 'draw' && shapeMode !== null) { // Şekil çizim
         drawCtx.clearRect(0, 0, drawCanvas.width, drawCanvas.height); // Önceki geçici şekli sil
         // Geçmişteki çizimleri yeniden çiz (sadece şekil çizim modundayken)
         if (drawHistory.length > 0) {
              drawCtx.putImageData(drawHistory[drawHistory.length - 1], 0, 0);
         }

         drawCtx.strokeStyle = brushColor;
         drawCtx.lineWidth = brushSize;
         drawCtx.shadowBlur = (brushType === 'glow') ? 10 : 0;
         drawCtx.shadowColor = (brushType === 'glow') ? brushColor : 'transparent';

         if (shapeMode === 'rect') {
             drawCtx.strokeRect(startShape.x, startShape.y, x - startShape.x, y - startShape.y);
         } else if (shapeMode === 'circle') {
             drawCtx.beginPath();
             // Merkez ve yarıçapı hesapla (startShape ve mevcut nokta)
             const centerX = (startShape.x + x) / 2;
             const centerY = (startShape.y + y) / 2;
             const radius = Math.sqrt(Math.pow(x - startShape.x, 2) + Math.pow(y - startShape.y, 2)) / 2;
             drawCtx.arc(centerX, centerY, radius, 0, 2 * Math.PI);
             drawCtx.stroke();
         }
    }
}

function handleMouseUp(e) {
    if (isDrawing && drawMode === 'draw') {
        isDrawing = false;
        if (shapeMode !== null) { // Şekil çizimi bittiğinde son halini geçmişe kaydet
             saveDrawState();
             shapeMode = null; // Şekil modunu sıfırla
             startShape = null;
        }
    }
}

function handleMouseOut() {
    isDrawing = false;
    // Eğer şekil çiziyorken mouse dışarı çıkarsa, şekli iptal etme veya son halini çizme kararı burada verilebilir.
    // Şimdilik bir şey yapmıyoruz.
}

// Canvas Click İşleyici (Yazı Ekleme)
function handleCanvasClick(e) {
    if (drawMode === 'text' && textInput.value.trim() !== '') {
         textToAdd = textInput.value.trim();
        saveDrawState(); // Yazı eklemeden önceki durumu kaydet
        const rect = drawCanvas.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;

        drawCtx.fillStyle = brushColor;
        drawCtx.font = brushSize * 3 + 'px Arial'; // Yazı boyutu fırça boyutuna göre ayarlanabilir
        drawCtx.fillText(textToAdd, x, y + (brushSize*3)/3); // Yazıyı tıklanan yere çiz
        // Not: Font boyutu ve baseline ayarlaması gerekebilir

        // textInput.value = ''; // Yazı eklendikten sonra inputu temizleme (kullanıcı aynı yazıyı tekrar eklemek isteyebilir)
        // textToAdd = '';
         saveDrawState(); // Yazı eklendikten sonra yeni durumu kaydet
    }
}

// Kaydet Butonu İşleyici
function handleSaveClick() {
    if (!image) { // image objesinin varlığını kontrol et, dosya adından daha güvenilir
        alert("Lütfen önce bir görsel yükleyin.");
        return;
    }

    showLoading('Görsel işleniyor...');

    // Sadece çizim canvas'ının içeriğini Data URL olarak al
    const drawDataUrl = drawCanvas.toDataURL('image/png');

    const formData = new FormData();
    // Orijinal görsel dosyasını FormData'ya ekle
    // handleFileSelect içinde global bir 'file' değişkeninde sakladığımızı varsayalım ya da inputtan tekrar alalım.
    // Daha güvenilir olması için fileInput'tan tekrar alıyorum:
    const originalFile = fileInput.files[0];
    if (!originalFile) {
         alert("Orijinal dosya bulunamadı.");
         hideLoading();
         return;
    }
    formData.append('original_file', originalFile, filename || 'image.png'); // Dosya adı korunuyor

    // Çizim verisini (Data URL) FormData'ya ekle
    formData.append('draw_data_url', drawDataUrl);

    $.ajax({
        url: '/draw_process_image', // app.py'deki '/draw_process_image' route'u
        type: 'POST',
        data: formData,
        processData: false,
        contentType: false,
        success: function(res){
            hideLoading();
            if(res.success){
                resultImage.src = res.result_url;
                imageDownloadBtn.href = res.result_url;
                imageDownloadBtn.download = filename ? 'processed_' + filename : 'processed_image.png'; // İndirme dosya adını ayarla
                resultContainer.style.display = 'block'; // Sonuç alanını göster
                canvasContainer.style.display = 'none'; // Canvas'ı gizle
                toolbar.style.display = 'none'; // Araç çubuğunu gizle
                 image = null; // Yeni düzenleme için görsel objesini sıfırla
            } else {
                alert('İşlem Hatası: '+res.error);
            }
        },
        error: function(xhr, status, error) {
            hideLoading();
            console.error('Ajax Hatası:', xhr);
            alert('Görsel işlenirken bir hata oluştu: ' + error);
        }
    });
}

// Data URL'yi Blob'a çevirme fonksiyonu (Artık kullanılmıyor, direkt file objesi gönderiliyor)

// Sayfayı Sıfırlama (Yeni Düzenleme)
function resetPage() {
     // Tüm alanları başlangıç durumuna getir
     filename = null;
     image = null; // Görsel objesini sıfırla
     drawHistory = [];
     drawCtx.clearRect(0, 0, drawCanvas.width, drawCanvas.height);
     imgCtx.clearRect(0, 0, mainCanvas.width, mainCanvas.height);

     uploadArea.style.display = 'block';
     canvasContainer.style.display = 'none';
     toolbar.style.display = 'none';
     resultContainer.style.display = 'none';
     fileInput.value = ''; // Dosya seçimini temizle
     // $('#uploaded-image').attr('src', ''); // Önizleme resmini temizle (artık kullanılmıyor)
     // $('#uploadBtn').hide(); // Yükle butonunu gizle (artık kullanılmıyor)
     setDrawMode('draw'); // Çizim moduna dön
     brushSizeControl.value = 5;
     brushSizeValueSpan.textContent = '5px';
     colorPicker.value = '#ff0000';
     brushTypeSelect.value = 'normal';
     textInput.value = '';
     textToAdd = '';
     shapeMode = null;
     startShape = null;
     isShift = false;
}

// Loading Spinner Fonksiyonları
function showLoading(message = 'Yükleniyor...') {
    $('#loading p').text(message); // Loading mesajını güncelle
    loadingSpinner.style.display = 'flex';
}

function hideLoading() {
    loadingSpinner.style.display = 'none';
}

// Sayfa yüklendiğinde...
$(document).ready(function() {
    // Başlangıç durumunu ayarla
    canvasContainer.style.display = 'none';
    toolbar.style.display = 'none';
    resultContainer.style.display = 'none';
    // $('#uploadBtn').hide(); // Başlangıçta yükle butonunu gizle (artık kullanılmıyor)
    setDrawMode('draw'); // Varsayılan mod çizim
}); 