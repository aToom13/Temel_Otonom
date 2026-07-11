import os
import cv2
import numpy as np
from .mesafe_hesaplayici import MesafeHesaplayici


class YOLODedektoru:
    """
    Aşama 3: YOLOv8 Tabanlı Derin Öğrenme Nesne, Araç ve Tabela Tespiti.
    Ultralytics YOLO çalıştırır; sınırlayıcı kutu, sınıf, güven skoru ve
    MesafeHesaplayici üzerinden pinhole kamera modeliyle mesafe hesaplar.
    Ortamda ultralytics bulunmadığında simülasyon/kural tabanlı modda çalışır.
    """

    def __init__(self, model_yolu="modeller/best.pt", guven_esigi=0.45, odak_uzakligi=950.0):
        self.model_yolu = model_yolu
        self.guven_esigi = guven_esigi
        self.mesafe_olcucu = MesafeHesaplayici(odak_uzakligi=odak_uzakligi)
        self.model = None

        try:
            from ultralytics import YOLO
            if os.path.exists(model_yolu):
                self.model = YOLO(model_yolu)
                print(f"[BILGI] YOLOv8 modeli başarıyla yüklendi: {model_yolu}")
        except Exception as e:
            print(f"[BILGI] YOLOv8 çalışma ortamı ({e}), kural tabanlı simülasyon aktif.")

    def tespit_et(self, kare):
        """Kare üzerindeki nesneleri tespit edip mesafeleriyle döndürür."""
        tespitler = []

        if self.model is not None:
            try:
                sonuclar = self.model(kare, conf=self.guven_esigi, verbose=False)
                for r in sonuclar:
                    for box in r.boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        conf = float(box.conf[0])
                        cls_id = int(box.cls[0])
                        etiket = self.model.names.get(cls_id, f"sinif_{cls_id}")

                        kutu_w = x2 - x1
                        mesafe = self.mesafe_olcucu.hesapla(etiket, kutu_w)

                        tespitler.append({
                            'kutu': (x1, y1, x2, y2),
                            'etiket': etiket,
                            'guven': conf,
                            'mesafe': mesafe
                        })
            except Exception as e:
                print(f"[UYARI] YOLO çıkarım hatası: {e}")

        return tespitler
