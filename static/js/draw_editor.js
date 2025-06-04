let fileType = '{{ file_type }}';
let image = null;
let videoElement = document.getElementById('videoElement');
let drawing = false;
let drawMode = 'draw';
let brushColor = '#ff0000';
let brushSize = 5;
let lastX = 0, lastY = 0;
let drawHistory = [];
let textToAdd = '';
let brushType = 'normal';
let shapeMode = null; // 'rect' veya 'circle'
let startShape = null;
let isShift = false;

const fileInput = document.getElementById('fileInput');
const mainCanvas = document.getElementById('mainCanvas');
const drawCanvas = document.getElementById('drawCanvas');
const canvasWrap = document.getElementById('canvasWrap');
const ctx = mainCanvas.getContext('2d');
const drawCtx = drawCanvas.getContext('2d');

fileInput.addEventListener('change', function(e) {
    const file = e.target.files[0];
    if (!file) return;
    const url = URL.createObjectURL(file);
    
    ctx.clearRect(0, 0, mainCanvas.width, mainCanvas.height);
    drawCtx.clearRect(0, 0, drawCanvas.width, drawCanvas.height);
    $('#resultContainer').hide();
    $('#videoElement').hide();
    drawHistory = [];

    if (fileType === 'image') {
        image = new Image();
        image.onload = function() {
            let maxW = 800, maxH = 500;
            let scale = Math.min(maxW / image.width, maxH / image.height, 1);
            let w = Math.round(image.width * scale);
            let h = Math.round(image.height * scale);
            mainCanvas.width = w;
            mainCanvas.height = h;
            drawCanvas.width = w;
            drawCanvas.height = h;
            ctx.drawImage(image, 0, 0, w, h);
            canvasWrap.style.display = 'flex';
            $(".video-controls").hide();
        };
        image.src = url;
    } else if (fileType === 'video') {
        videoElement.style.display = 'block';
        videoElement.src = url;
        videoElement.crossOrigin = "anonymous";
        videoElement.onloadedmetadata = function() {
            let maxW = 800, maxH = 500;
            let scale = Math.min(maxW / videoElement.videoWidth, maxH / videoElement.videoHeight, 1);
            let w = Math.round(videoElement.videoWidth * scale);
            let h = Math.round(videoElement.videoHeight * scale);
            mainCanvas.width = w;
            mainCanvas.height = h;
            drawCanvas.width = w;
            drawCanvas.height = h;
            canvasWrap.style.display = 'flex';
            $(".video-controls").show();
            $("#duration").text(formatTime(videoElement.duration));
        };
        videoElement.ontimeupdate = function() {
            $("#currentTime").text(formatTime(videoElement.currentTime));
            $("#videoSeek").val((videoElement.currentTime / videoElement.duration) * 100);
            ctx.drawImage(videoElement, 0, 0, mainCanvas.width, mainCanvas.height);
        };
    }
});

function formatTime(sec) {
    sec = Math.floor(sec);
    return Math.floor(sec/60) + ':' + ('0'+(sec%60)).slice(-2);
}

$('#drawBtn').click(function(){ drawMode = 'draw'; $(this).addClass('active'); $('#textBtn').removeClass('active'); $('#textInputWrap').hide(); });
$('#textBtn').click(function(){ drawMode = 'text'; $(this).addClass('active'); $('#drawBtn').removeClass('active'); $('#textInputWrap').show(); });
$('#colorPicker').on('input', function(){ brushColor = this.value; });
$('#brushSize').on('input', function(){ brushSize = this.value; $('#brushSizeValue').text(brushSize + 'px'); });
$('#clearBtn').click(function(){ drawCtx.clearRect(0, 0, drawCanvas.width, drawCanvas.height); drawHistory = []; });
$('#resetBtn').click(function(){
    ctx.clearRect(0, 0, mainCanvas.width, mainCanvas.height);
    drawCtx.clearRect(0, 0, drawCanvas.width, drawCanvas.height);
    if(fileType==='image' && image) ctx.drawImage(image, 0, 0, mainCanvas.width, mainCanvas.height);
    if(fileType==='video' && videoElement) ctx.drawImage(videoElement, 0, 0, mainCanvas.width, mainCanvas.height);
    drawHistory = [];
});
$('#undoBtn').click(function(){ if(drawHistory.length>0){ drawCtx.putImageData(drawHistory.pop(),0,0); } });
$('#textInput').on('input', function(){ textToAdd = this.value; });
$('#brushType').on('change', function(){ brushType = this.value; });
$('#rectBtn').click(function(){ shapeMode = 'rect'; });
$('#circleBtn').click(function(){ shapeMode = 'circle'; });
$(document).keydown(function(e){ if(e.key==='Shift') isShift = true; });
$(document).keyup(function(e){ if(e.key==='Shift') isShift = false; });

function saveDrawHistory() {
    drawHistory.push(drawCtx.getImageData(0,0,drawCanvas.width,drawCanvas.height));
    if(drawHistory.length>20) drawHistory.shift();
}
drawCanvas.addEventListener('mousedown', function(e){
    if(drawMode==='draw') {
        drawing = true;
        saveDrawHistory();
        const rect = drawCanvas.getBoundingClientRect();
        lastX = e.clientX - rect.left;
        lastY = e.clientY - rect.top;
        if(shapeMode) {
            startShape = {x: lastX, y: lastY};
        }
    }
});
drawCanvas.addEventListener('mousemove', function(e){
    if(drawing && drawMode==='draw') {
        const rect = drawCanvas.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;
        drawCtx.strokeStyle = brushColor;
        drawCtx.lineWidth = brushSize;
        drawCtx.lineCap = 'round';
        drawCtx.shadowBlur = (brushType==='glow') ? 10 : 0;
        drawCtx.shadowColor = (brushType==='glow') ? brushColor : 'transparent';
        if(shapeMode) {
            drawCtx.clearRect(0, 0, drawCanvas.width, drawCanvas.height);
            if(shapeMode==='rect') {
                drawCtx.strokeRect(startShape.x, startShape.y, x-startShape.x, y-startShape.y);
            } else if(shapeMode==='circle') {
                drawCtx.beginPath();
                let r = Math.sqrt(Math.pow(x-startShape.x,2)+Math.pow(y-startShape.y,2));
                drawCtx.arc(startShape.x, startShape.y, r, 0, 2*Math.PI);
                drawCtx.stroke();
            }
        } else if(isShift) {
            drawCtx.beginPath();
            drawCtx.moveTo(lastX, lastY);
            drawCtx.lineTo(x, y);
            drawCtx.stroke();
        } else if(brushType==='dotted') {
            drawCtx.beginPath();
            drawCtx.setLineDash([5, 10]);
            drawCtx.moveTo(lastX, lastY);
            drawCtx.lineTo(x, y);
            drawCtx.stroke();
            drawCtx.setLineDash([]);
            lastX = x; lastY = y;
        } else {
            drawCtx.beginPath();
            drawCtx.moveTo(lastX, lastY);
            drawCtx.lineTo(x, y);
            drawCtx.stroke();
            lastX = x; lastY = y;
        }
    }
});
drawCanvas.addEventListener('mouseup', function(e){
    if(shapeMode && drawing) {
        const rect = drawCanvas.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;
        drawCtx.strokeStyle = brushColor;
        drawCtx.lineWidth = brushSize;
        drawCtx.shadowBlur = (brushType==='glow') ? 10 : 0;
        drawCtx.shadowColor = (brushType==='glow') ? brushColor : 'transparent';
        if(shapeMode==='rect') {
            drawCtx.strokeRect(startShape.x, startShape.y, x-startShape.x, y-startShape.y);
        } else if(shapeMode==='circle') {
            drawCtx.beginPath();
            let r = Math.sqrt(Math.pow(x-startShape.x,2)+Math.pow(y-startShape.y,2));
            drawCtx.arc(startShape.x, startShape.y, r, 0, 2*Math.PI);
            drawCtx.stroke();
        }
        shapeMode = null;
        startShape = null;
    }
    drawing = false;
});
drawCanvas.addEventListener('mouseout', function(){ drawing = false; });
drawCanvas.addEventListener('click', function(e){
    if(drawMode==='text' && textToAdd) {
        saveDrawHistory();
        const rect = drawCanvas.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;
        drawCtx.fillStyle = brushColor;
        drawCtx.font = brushSize*3 + 'px Arial';
        drawCtx.fillText(textToAdd, x, y);
    }
});

document.getElementById('playPauseBtn').onclick = function(){
    if(videoElement.paused) videoElement.play(); else videoElement.pause();
};
document.getElementById('videoSeek').oninput = function(){
    if(videoElement) videoElement.currentTime = (this.value/100)*videoElement.duration;
};

// İndirme butonu (canvas'ı indir)
document.getElementById('downloadBtn').onclick = function(){
    const merged = document.createElement('canvas');
    merged.width = mainCanvas.width;
    merged.height = mainCanvas.height;
    const mctx = merged.getContext('2d');
    mctx.drawImage(mainCanvas,0,0);
    mctx.drawImage(drawCanvas,0,0);
    const link = document.createElement('a');
    link.download = 'duzenlenmis_gorsel.png';
    link.href = merged.toDataURL('image/png');
    link.click();
};

// Geri dön butonu
$('#backBtn').click(function(){ window.location.href = '/'; });

document.getElementById('saveBtn').onclick = function(){
    if(fileType === 'image') {
        const merged = document.createElement('canvas');
        merged.width = mainCanvas.width;
        merged.height = mainCanvas.height;
        const mctx = merged.getContext('2d');
        mctx.drawImage(mainCanvas,0,0);
        mctx.drawImage(drawCanvas,0,0);
        merged.toBlob(function(blob){
            const formData = new FormData();
            formData.append('file', fileInput.files[0]);
            formData.append('draw', merged.toDataURL('image/png'));
            $.ajax({
                url: '/draw_process_image',
                type: 'POST',
                data: formData,
                processData: false,
                contentType: false,
                success: function(res){
                    if(res.success){
                        $('#resultImage').attr('src', res.result_url).show();
                        $('#resultVideo').hide();
                        $('#videoDownloadBtn').hide();
                        $('#downloadBtn').attr('href', res.result_url).attr('download', 'duzenlenmis_gorsel.png').show();
                        $('#resultContainer').show();
                    } else {
                        alert('Hata: '+res.error);
                    }
                }
            });
        },'image/png');
    } else if(fileType === 'video') {
        const drawDataUrl = drawCanvas.toDataURL('image/png');
        const formData = new FormData();
        formData.append('file', fileInput.files[0]);
        formData.append('draw', drawDataUrl);
        $.ajax({
            url: '/draw_process_video',
            type: 'POST',
            data: formData,
            processData: false,
            contentType: false,
            success: function(res){
                if(res.success){
                    $('#resultVideo').attr('src', res.result_url).show();
                    $('#resultImage').hide();
                    $('#videoDownloadBtn').attr('href', res.result_url).attr('download', 'duzenlenmis_video.mp4').show();
                    $('#downloadBtn').hide();
                    $('#resultContainer').show();
                } else {
                    alert('Hata: '+res.error);
                }
            }
        });
    }
}; 