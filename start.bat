@echo off
echo ZITS Resim Düzenleme Uygulaması başlatılıyor...
echo.

rem Python ortamını kontrol et
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo HATA: Python yüklü değil veya PATH'te değil.
    echo Lütfen Python 3.8 veya üstünü yükleyin.
    pause
    exit /b 1
)

rem ZITS klasörünü kontrol et
if not exist ZITS (
    echo HATA: ZITS klasörü bulunamadı.
    echo Lütfen önce ZITS modelini yükleyin:
    echo git clone https://github.com/JingyunLiang/ZITS.git
    pause
    exit /b 1
)

rem Flask uygulamasını başlat
python app/app.py

pause 