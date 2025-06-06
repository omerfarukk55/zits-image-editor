$(document).ready(function() {
    // Global değişkenler
    let videoElement, mainCanvas, drawCanvas, ctx, drawCtx;
    let isDrawing = false;
    let drawMode = 'draw';
    let brushColor = '#ff0000';
    let brushSize = 5;
    let lastX = 0, lastY = 0;
    let brushType = 'normal';
    let shapeMode = null;
    let startShape = null;
    let isShift = false;
    let filename = null;
    let drawHistory = [];

    // DOM elementlerini al
    const $videoElement = $('#videoElement');
    const $mainCanvas = $('#mainCanvas');
    const $drawCanvas = $('#drawCanvas');
    const $canvasContainer = $('#canvas-container');
    const $uploadArea = $('#upload-area');
    const $toolbar = $('.toolbar');
    const $resultContainer = $('#resultContainer');
    const $loading = $('#loading');

    // Canvas context'leri al
    videoElement = $videoElement[0];
    mainCanvas = $mainCanvas[0];
    drawCanvas = $drawCanvas[0];
    ctx = mainCanvas?.getContext('2d');
    drawCtx = drawCanvas?.getContext('2d');

    // Event Listeners
    $('#fileInput').on('change', handleFileSelect);
    $('#playPauseBtn').on('click', togglePlayPause);
    $('#videoSeek').on('input', seekVideo);
    $('#drawBtn').on('click', () => setDrawMode('draw'));
    $('#textBtn').on('click', () => setDrawMode('text'));
    $('#colorPicker').on('input', (e) => brushColor = e.target.value);
    $('#brushSize').on('input', handleBrushSizeChange);
    $('#brushType').on('change', (e) => brushType = e.target.value);
    $('#rectBtn').on('click', () => setShapeMode('rect'));
    $('#circleBtn').on('click', () => setShapeMode('circle'));
    $('#clearBtn').on('click', clearDrawCanvas);
    $('#resetBtn').on('click', resetCanvas);
    $('#undoBtn').on('click', undoLastDraw);
    $('#saveBtn').on('click', handleSaveClick);
    $('#newEditBtn').on('click', resetPage);

    // Canvas mouse events
    $drawCanvas.on({
        mousedown: handleMouseDown,
        mousemove: handleMouseMove,
        mouseup: handleMouseUp,
        mouseout: () => isDrawing = false,
        click: handleCanvasClick
    });

    // Keyboard events
    $(document).on('keydown keyup', function(e) {
        isShift = e.shiftKey;
        
        if (e.type === 'keydown') {
            if (e.ctrlKey && e.key === 'z') {
                e.preventDefault();
                undoLastDraw();
            }
            if (e.key === 'Escape') setDrawMode('draw');
            if (e.code === 'Space' && videoElement.src) {
                e.preventDefault();
                togglePlayPause();
            }
        }
    });

    // Drag & Drop
    $uploadArea.on({
        dragover: (e) => {
            e.preventDefault();
            $uploadArea.css('background-color', '#e3f2fd');
        },
        dragleave: (e) => {
            e.preventDefault();
            $uploadArea.css('background-color', '#f8f9fa');
        },
        drop: (e) => {
            e.preventDefault();
            $uploadArea.css('background-color', '#f8f9fa');
            const files = e.originalEvent.dataTransfer.files;
            if (files.length && files[0].type.startsWith('video/')) {
                $('#fileInput')[0].files = files;
                handleFileSelect({ target: { files } });
            }
        }
    });

    // Touch events for mobile
    $drawCanvas.on({
        touchstart: (e) => {
            e.preventDefault();
            const touch = e.originalEvent.touches[0];
            const rect = drawCanvas.getBoundingClientRect();
            const mockEvent = {
                clientX: touch.clientX,
                clientY: touch.clientY
            };
            handleMouseDown(mockEvent);
        },
        touchmove: (e) => {
            e.preventDefault();
            const touch = e.originalEvent.touches[0];
            const mockEvent = {
                clientX: touch.clientX,
                clientY: touch.clientY
            };
            handleMouseMove(mockEvent);
        },
        touchend: (e) => {
            e.preventDefault();
            handleMouseUp();
        }
    });

    // Resize handler
    $(window).on('resize', debounce(resizeCanvas, 250));

    // Functions
    function handleFileSelect(e) {
        const file = e.target.files[0];
        if (!file) return resetPage();

        if (!file.type.startsWith('video/')) {
            alert("🚫 Lütfen bir video dosyası seçin.");
            $('#fileInput').val('');
            return resetPage();
        }

        // Dosya boyutu kontrolü (100MB)
        if (file.size > 100 * 1024 * 1024) {
            alert("🚫 Dosya boyutu çok büyük. Maksimum 100MB desteklenir.");
            $('#fileInput').val('');
            return resetPage();
        }

        showLoading('📹 Video yükleniyor...');
        filename = file.name;
        
        const fileURL = URL.createObjectURL(file);
        $videoElement.attr('src', fileURL);

        $videoElement.on('loadedmetadata', function() {
            hideLoading();
            setupCanvas();
            setupVideoEvents();
            initializeVideo();
            
            setTimeout(drawVideoFrame, 100);
            
            $uploadArea.hide();
            $canvasContainer.show();
            $toolbar.show();
            
            setDrawMode('draw');
            clearDrawCanvas();
            
            // Kullanıcıya bilgi ver
            showSuccessMessage('✅ Video yüklendi! Şimdi takip etmek istediğiniz nesnenin üzerine çizim yapın.');
        }).on('error', function() {
            hideLoading();
            alert("🚫 Video yüklenirken bir hata oluştu.");
            resetPage();
        });
    }

    function setupCanvas() {
        const containerWidth = $canvasContainer.width() || 800;
        const videoWidth = videoElement.videoWidth || 800;
        const videoHeight = videoElement.videoHeight || 450;
        const aspectRatio = videoHeight / videoWidth;
        const containerHeight = Math.min(containerWidth * aspectRatio, 500);
        
        $canvasContainer.height(containerHeight);
        
        $mainCanvas.add($drawCanvas).attr({
            width: containerWidth,
            height: containerHeight
        });

        if (ctx) {
            ctx.imageSmoothingEnabled = true;
            ctx.imageSmoothingQuality = 'high';
        }
        if (drawCtx) {
            drawCtx.lineCap = 'round';
            drawCtx.lineJoin = 'round';
            drawCtx.imageSmoothingEnabled = true;
            drawCtx.imageSmoothingQuality = 'high';
        }
    }

    function setupVideoEvents() {
        $videoElement.off('.videoEvents').on('timeupdate.videoEvents', function() {
            drawVideoFrame();
            updateVideoTime();
        }).on('play.videoEvents pause.videoEvents ended.videoEvents', updatePlayPauseButton);
    }

    function drawVideoFrame() {
        if (!videoElement.videoWidth || !mainCanvas.width) return;
        
        ctx.clearRect(0, 0, mainCanvas.width, mainCanvas.height);
        
        const videoAspectRatio = videoElement.videoWidth / videoElement.videoHeight;
        const canvasAspectRatio = mainCanvas.width / mainCanvas.height;
        
        let renderWidth, renderHeight, offsetX, offsetY;
        
        if (videoAspectRatio > canvasAspectRatio) {
            renderWidth = mainCanvas.width;
            renderHeight = mainCanvas.width / videoAspectRatio;
            offsetX = 0;
            offsetY = (mainCanvas.height - renderHeight) / 2;
        } else {
            renderHeight = mainCanvas.height;
            renderWidth = mainCanvas.height * videoAspectRatio;
            offsetX = (mainCanvas.width - renderWidth) / 2;
            offsetY = 0;
        }
        
        ctx.drawImage(videoElement, offsetX, offsetY, renderWidth, renderHeight);
    }

    function initializeVideo() {
        const duration = videoElement.duration || 0;
        $('#duration').text(formatTime(duration));
        $('#videoSeek').attr('max', duration).val(0);
    }

    function updatePlayPauseButton() {
        const icon = (videoElement.paused || videoElement.ended) ? 'play' : 'pause';
        $('#playPauseBtn').html(`<i class="fas fa-${icon}"></i>`);
    }

    function updateVideoTime() {
        const currentTime = videoElement.currentTime || 0;
        $('#currentTime').text(formatTime(currentTime));
        $('#videoSeek').val(currentTime);
    }

    function seekVideo() {
        videoElement.currentTime = $('#videoSeek').val();
    }

    function togglePlayPause() {
        videoElement.paused ? videoElement.play() : videoElement.pause();
    }

    function setDrawMode(mode) {
        drawMode = mode;
        shapeMode = null;
        
        $('.toolbar button').removeClass('active');
        
        if (mode === 'draw') {
            $('#drawBtn').addClass('active');
            $('#textInputWrap').hide();
            $drawCanvas.css('cursor', 'crosshair');
        } else if (mode === 'text') {
            $('#textBtn').addClass('active');
            $('#textInputWrap').show();
            $drawCanvas.css('cursor', 'text');
        }
    }

    function setShapeMode(mode) {
        setDrawMode('draw');
        shapeMode = mode;
        $('#rectBtn, #circleBtn').removeClass('active');
        $(`#${mode}Btn`).addClass('active');
    }

    function handleBrushSizeChange(e) {
        brushSize = parseInt(e.target.value);
        $('#brushSizeValue').text(brushSize + 'px');
    }

    function saveDrawState() {
        if (!drawCtx || !drawCanvas.width) return;
        
        try {
            const imageData = drawCtx.getImageData(0, 0, drawCanvas.width, drawCanvas.height);
            drawHistory.push(imageData);
            if (drawHistory.length > 20) drawHistory.shift();
        } catch (error) {
            console.error('Çizim durumu kaydetme hatası:', error);
        }
    }

    function handleMouseDown(e) {
        const rect = drawCanvas.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;
        
        if (drawMode === 'draw') {
            isDrawing = true;
            saveDrawState();
            
            if (shapeMode) {
                startShape = { x, y };
            } else {
                lastX = x;
                lastY = y;
            }
        }
    }

    function handleMouseMove(e) {
        if (!isDrawing || drawMode !== 'draw' || !drawCtx) return;
        
        const rect = drawCanvas.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;
        
        drawCtx.strokeStyle = brushColor;
        drawCtx.lineWidth = brushSize;
        drawCtx.lineCap = 'round';
        drawCtx.lineJoin = 'round';
        drawCtx.shadowBlur = brushType === 'glow' ? 15 : 0;
        drawCtx.shadowColor = brushType === 'glow' ? brushColor : 'transparent';
        
        if (!shapeMode) {
            // Serbest çizim
            drawCtx.beginPath();
            if (brushType === 'dotted') drawCtx.setLineDash([5, 10]);
            drawCtx.moveTo(lastX, lastY);
            drawCtx.lineTo(x, y);
            drawCtx.stroke();
            if (brushType === 'dotted') drawCtx.setLineDash([]);
            [lastX, lastY] = [x, y];
        } else {
            // Şekil çizimi
            if (drawHistory.length) {
                drawCtx.putImageData(drawHistory[drawHistory.length - 1], 0, 0);
            } else {
                drawCtx.clearRect(0, 0, drawCanvas.width, drawCanvas.height);
            }
            
            drawCtx.beginPath();
            if (shapeMode === 'rect') {
                drawCtx.strokeRect(startShape.x, startShape.y, x - startShape.x, y - startShape.y);
            } else if (shapeMode === 'circle') {
                const radius = Math.sqrt((x - startShape.x) ** 2 + (y - startShape.y) ** 2);
                drawCtx.arc(startShape.x, startShape.y, radius, 0, 2 * Math.PI);
                drawCtx.stroke();
            }
        }
    }

    function handleMouseUp() {
        if (isDrawing && drawMode === 'draw') {
            isDrawing = false;
            if (shapeMode) {
                saveDrawState();
                shapeMode = null;
                startShape = null;
                $('#rectBtn, #circleBtn').removeClass('active');
            }
        }
    }

    function handleCanvasClick(e) {
        if (drawMode === 'text') {
            const text = $('#textInput').val().trim();
            if (!text) return;
            
            saveDrawState();
            const rect = drawCanvas.getBoundingClientRect();
            const x = e.clientX - rect.left;
            const y = e.clientY - rect.top;
            
            drawCtx.fillStyle = brushColor;
            drawCtx.font = `bold ${brushSize * 3}px Arial`;
            drawCtx.textBaseline = 'top';
            drawCtx.fillText(text, x, y);
            saveDrawState();
        }
    }

    function clearDrawCanvas() {
        if (confirm('🗑️ Tüm çizimleri silmek istediğinizden emin misiniz?')) {
            drawCtx?.clearRect(0, 0, drawCanvas.width, drawCanvas.height);
            drawHistory = [];
        }
    }

    function resetCanvas() {
        if (!videoElement.src) return alert("⚠️ Lütfen önce bir video yükleyin.");
        
        videoElement.currentTime = 0;
        $('#videoSeek').val(0);
        videoElement.pause();
        setTimeout(() => {
            drawVideoFrame();
            clearDrawCanvas();
        }, 100);
    }

    function undoLastDraw() {
        if (drawHistory.length > 1) {
            drawHistory.pop();
            const lastState = drawHistory[drawHistory.length - 1];
            drawCtx.clearRect(0, 0, drawCanvas.width, drawCanvas.height);
            drawCtx.putImageData(lastState, 0, 0);
        } else if (drawHistory.length === 1) {
            drawHistory = [];
            drawCtx.clearRect(0, 0, drawCanvas.width, drawCanvas.height);
        }
    }

    function handleSaveClick() {
        if (!videoElement.src) return alert("⚠️ Lütfen önce bir video yükleyin.");

        const originalFile = $('#fileInput')[0].files[0];
        if (!originalFile) return alert("⚠️ Orijinal dosya bulunamadı.");

        const drawDataUrl = drawCanvas.toDataURL('image/png');
        const isCanvasEmpty = isDrawCanvasEmpty(drawDataUrl);
        
        if (isCanvasEmpty) {
            return alert("⚠️ Lütfen takip etmek istediğiniz nesnenin üzerine çizim yapın.");
        }

        if (!confirm('🎯 Akıllı nesne takibi başlatılacak. Bu işlem birkaç dakika sürebilir. Devam etmek istiyor musunuz?')) {
            return;
        }

        showProcessingAnimation();

        const formData = new FormData();
        formData.append('video_file', originalFile, filename || 'video.mp4');
        formData.append('draw_mask_url', drawDataUrl);

        // Progress simulation
        let progress = 0;
        const progressInterval = setInterval(() => {
            progress += Math.random() * 3;
            if (progress > 90) progress = 90;
            updateProgress(progress);
        }, 1000);

        $.ajax({
            url: '/draw_process_video',
            type: 'POST',
            data: formData,
            processData: false,
            contentType: false,
            timeout: 600000, // 10 dakika
            success: function(res) {
                clearInterval(progressInterval);
                updateProgress(100);
                
                setTimeout(() => {
                    hideLoading();
                    
                    if (res.success) {
                        $('#resultVideo').attr('src', res.result_url).show();
                        $('#videoDownloadBtn').attr({
                            href: res.result_url,
                            download: filename ? 'tracked_' + filename : 'tracked_video.mp4'
                        });
                        $resultContainer.show();
                        $canvasContainer.hide();
                        $toolbar.hide();
                        
                        // Başarı bildirimi
                        showSuccessMessage('🎉 Nesne takibi başarıyla tamamlandı! Videonu indirebilirsiniz.');
                    } else {
                        alert('❌ İşlem Hatası: ' + (res.error || 'Bilinmeyen hata'));
                    }
                }, 1000);
            },
            error: function(xhr, status, error) {
                clearInterval(progressInterval);
                hideLoading();
                
                let errorMessage = '❌ Video işlenirken bir hata oluştu.';
                if (status === 'timeout') {
                    errorMessage = '⏰ İşlem zaman aşımına uğradı. Lütfen daha küçük bir video deneyin.';
                } else if (xhr.responseJSON?.error) {
                    errorMessage = '❌ ' + xhr.responseJSON.error;
                }
                alert(errorMessage);
            }
        });
    }

    function isDrawCanvasEmpty(dataUrl) {
        // Boş canvas'ın data URL'si
        const emptyCanvas = document.createElement('canvas');
        emptyCanvas.width = drawCanvas.width;
        emptyCanvas.height = drawCanvas.height;
        const emptyDataUrl = emptyCanvas.toDataURL('image/png');
        
        return dataUrl === emptyDataUrl;
    }

    function showProcessingAnimation() {
        const messages = [
            '🎬 Video analiz ediliyor...',
            '🎯 Çizim bölgeleri tespit ediliyor...',
            '🧠 RAFT Optical Flow hesaplanıyor...',
            '🔄 Nesne takibi uygulanıyor...',
            '🎨 Video render ediliyor...',
            '⚡ Son rötuşlar yapılıyor...'
        ];
        
        let messageIndex = 0;
        showLoading(messages[messageIndex]);
        
        const messageInterval = setInterval(() => {
            messageIndex = (messageIndex + 1) % messages.length;
            $('#loadingText').text(messages[messageIndex]);
        }, 3000);
        
        // Interval'ı global'e kaydet ki daha sonra temizleyebilelim
        window.messageInterval = messageInterval;
    }

    function updateProgress(percentage) {
        $('#progressFill').css('width', percentage + '%');
    }

    function resetPage() {
        // Interval'ları temizle
        if (window.messageInterval) {
            clearInterval(window.messageInterval);
            window.messageInterval = null;
        }

        // Reset variables
        filename = null;
        drawHistory = [];
        isDrawing = false;
        drawMode = 'draw';
        shapeMode = null;
        startShape = null;
        isShift = false;

        // Clear canvases
        drawCtx?.clearRect(0, 0, drawCanvas.width, drawCanvas.height);
        ctx?.clearRect(0, 0, mainCanvas.width, mainCanvas.height);

        // Reset video
        if (videoElement.src) {
            URL.revokeObjectURL(videoElement.src);
        }
        $videoElement.removeAttr('src')[0].load();

        // Reset UI
        $uploadArea.show();
        $canvasContainer.hide();
        $toolbar.hide();
        $resultContainer.hide();

        // Reset form values
        $('#fileInput').val('');
        $('#brushSize').val(5);
        $('#brushSizeValue').text('5px');
        $('#colorPicker').val('#ff0000');
        $('#brushType').val('normal');
        $('#textInput').val('');
        $('#playPauseBtn').html('<i class="fas fa-play"></i>');
        $('#videoSeek').val(0);
        $('#currentTime, #duration').text('0:00');

        setDrawMode('draw');
        hideLoading();
        
        // Progress'i sıfırla
        updateProgress(0);
    }

    function resizeCanvas() {
        if ($canvasContainer.is(':hidden')) return;
        
        const imageData = drawCtx?.getImageData(0, 0, drawCanvas.width, drawCanvas.height);
        setupCanvas();
        
        if (imageData && drawCanvas.width > 0) {
            const tempCanvas = document.createElement('canvas');
            const tempCtx = tempCanvas.getContext('2d');
            tempCanvas.width = imageData.width;
            tempCanvas.height = imageData.height;
            tempCtx.putImageData(imageData, 0, 0);
            drawCtx.drawImage(tempCanvas, 0, 0, drawCanvas.width, drawCanvas.height);
        }
        
        drawVideoFrame();
    }

    // Utility functions
    function formatTime(sec) {
        sec = Math.floor(sec || 0);
        const minutes = Math.floor(sec / 60);
        const seconds = sec % 60;
        return `${minutes}:${seconds.toString().padStart(2, '0')}`;
    }

    function showLoading(message = 'Yükleniyor...') {
        $('#loadingText').text(message);
        $loading.show();
        updateProgress(0);
    }

    function hideLoading() {
        $loading.hide();
        
        // Interval'ları temizle
        if (window.messageInterval) {
            clearInterval(window.messageInterval);
            window.messageInterval = null;
        }
    }

    function showSuccessMessage(message) {
        // Toast notification
        const toast = $(`
            <div class="toast-notification" style="
                position: fixed;
                top: 20px;
                right: 20px;
                background: linear-gradient(45deg, #28a745, #20c997);
                color: white;
                padding: 15px 20px;
                border-radius: 10px;
                box-shadow: 0 4px 12px rgba(0,0,0,0.3);
                z-index: 9999;
                max-width: 300px;
                animation: slideIn 0.5s ease-out;
            ">
                ${message}
            </div>
        `);
        
        $('body').append(toast);
        
        setTimeout(() => {
            toast.fadeOut(500, () => toast.remove());
        }, 5000);
    }

    function debounce(func, wait) {
        let timeout;
        return function executedFunction(...args) {
            const later = () => {
                clearTimeout(timeout);
                func(...args);
            };
            clearTimeout(timeout);
            timeout = setTimeout(later, wait);
        };
    }

    // Keyboard shortcuts helper
    function showKeyboardShortcuts() {
        const shortcuts = `
            <div class="keyboard-shortcuts">
                <h6>⌨️ Klavye Kısayolları:</h6>
                <ul>
                    <li><kbd>Space</kbd> - Video Oynat/Durdur</li>
                    <li><kbd>Ctrl+Z</kbd> - Geri Al</li>
                    <li><kbd>Esc</kbd> - Çizim Moduna Geç</li>
                    <li><kbd>Shift</kbd> + Çizim - Düz Çizgi</li>
                </ul>
            </div>
        `;
        
        // Shortcuts'u toolbar'a ekle
        if (!$('.keyboard-shortcuts').length) {
            $toolbar.append(shortcuts);
        }
    }

    // Performance monitoring
    function monitorPerformance() {
        if (performance.memory) {
            console.log('Memory Usage:', {
                used: Math.round(performance.memory.usedJSHeapSize / 1048576) + 'MB',
                total: Math.round(performance.memory.totalJSHeapSize / 1048576) + 'MB',
                limit: Math.round(performance.memory.jsHeapSizeLimit / 1048576) + 'MB'
            });
        }
    }

    // Error handling
    window.addEventListener('error', function(e) {
        console.error('JavaScript Hatası:', e.error);
        hideLoading();
        
        if (e.error && e.error.message) {
            alert('❌ Bir hata oluştu: ' + e.error.message);
        }
    });

    // Unhandled promise rejections
    window.addEventListener('unhandledrejection', function(e) {
        console.error('Promise Hatası:', e.reason);
        hideLoading();
    });

    // CSS animasyonları ekle
    const style = document.createElement('style');
    style.textContent = `
        @keyframes slideIn {
            from {
                transform: translateX(100%);
                opacity: 0;
            }
            to {
                transform: translateX(0);
                opacity: 1;
            }
        }
        
        .toast-notification {
            font-weight: 500;
            font-size: 14px;
            line-height: 1.4;
        }
        
        .keyboard-shortcuts {
            background: #f8f9fa;
            padding: 10px;
            border-radius: 5px;
            margin-top: 10px;
            font-size: 12px;
        }
        
        .keyboard-shortcuts ul {
            margin: 5px 0 0 0;
            padding-left: 20px;
        }
        
        .keyboard-shortcuts li {
            margin: 2px 0;
        }
        
        kbd {
            background: #e9ecef;
            padding: 2px 4px;
            border-radius: 3px;
            font-size: 11px;
            font-weight: bold;
        }
        
        #drawCanvas:hover {
            cursor: crosshair;
        }
        
        .processing-info {
            animation: pulse 2s infinite;
        }
        
        @keyframes pulse {
            0% { opacity: 1; }
            50% { opacity: 0.7; }
            100% { opacity: 1; }
        }
        
        .progress-fill {
            background: linear-gradient(90deg, #007bff, #28a745, #ffc107);
            background-size: 200% 100%;
            animation: progressShine 2s linear infinite;
        }
        
        @keyframes progressShine {
            0% { background-position: 200% 0; }
            100% { background-position: -200% 0; }
        }
    `;
    document.head.appendChild(style);

    // Başlangıç durumunu ayarla
    resetPage();
    
    // Performance monitoring başlat (development için)
    if (window.location.hostname === 'localhost') {
        setInterval(monitorPerformance, 30000); // Her 30 saniyede bir
        showKeyboardShortcuts();
    }
    
    console.log('🎯 Akıllı Video Nesne Takip Sistemi başarıyla yüklendi!');
    console.log('📊 Özellikler: RAFT Optical Flow, Çoklu Bölge Takibi, Gerçek Zamanlı Rendering');
});