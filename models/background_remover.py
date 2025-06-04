from rembg import remove
import numpy as np
from PIL import Image
import io
import cv2

class BackgroundRemover:
    def __init__(self):
        self.model_loaded = False
        print("Arka plan kaldırma modeli başlatılıyor...")
        self.model_loaded = True
        print("Arka plan kaldırma modeli başarıyla başlatıldı")

    def remove_background(self, image):
        """
        Görüntüden arka planı kaldırır ve saydam yapar
        image: numpy array (BGR format)
        """
        try:
            # BGR'den RGB'ye dönüştür
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # NumPy array'den PIL Image'a dönüştür
            pil_image = Image.fromarray(rgb_image)
            
            # Arka planı kaldır
            output = remove(pil_image)
            
            # PIL Image'dan NumPy array'e dönüştür
            result = np.array(output)
            
            # RGBA formatında olduğu için BGR'ye dönüştürmeye gerek yok
            # Doğrudan RGBA formatında kaydet
            return result

        except Exception as e:
            print(f"Arka plan kaldırma hatası: {str(e)}")
            return image

def load_background_remover():
    """Arka plan kaldırma modelini yükle"""
    try:
        return BackgroundRemover()
    except Exception as e:
        print(f"Arka plan kaldırma modeli yüklenirken hata: {str(e)}")
        return None 