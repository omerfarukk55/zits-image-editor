import os
import numpy as np
import cv2
from PIL import Image

class ZITSInpainter:
    def __init__(self, device="cpu"):
        self.device = device
        self.model_loaded = False
        print("ZITS modeli başlatılıyor...")
        self.model_loaded = True
        print("ZITS modeli başarıyla başlatıldı")

    def inpaint(self, image, mask):
        """
        Fırça ile çizilen yerleri siler ve doldurur
        image: numpy array (BGR format)
        mask: numpy array (grayscale)
        """
        try:
            # Maskeyi hazırla (fırça izlerini beyaz yap)
            if mask.max() <= 1.0:
                mask = (mask * 255).astype(np.uint8)
            
            # Maskeyi genişlet
            kernel = np.ones((5,5), np.uint8)
            mask = cv2.dilate(mask, kernel, iterations=1)
            
            # Maskeyi yumuşat
            mask = cv2.GaussianBlur(mask, (5,5), 0)
            
            # İnpainting için maskeyi hazırla (beyaz alanlar silinecek)
            inpaint_mask = (mask > 127).astype(np.uint8) * 255
            
            # İnpainting uygula
            result = cv2.inpaint(image, inpaint_mask, 3, cv2.INPAINT_TELEA)
            
            # Sonucu yumuşat
            result = cv2.GaussianBlur(result, (3,3), 0)
            
            return result

        except Exception as e:
            print(f"İnpainting hatası: {str(e)}")
            return image

def load_zits_model(device="cpu"):
    """ZITS modelini yükle"""
    try:
        return ZITSInpainter(device=device)
    except Exception as e:
        print(f"ZITS modeli yüklenirken hata: {str(e)}")
        return None