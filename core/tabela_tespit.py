import os


class TabelaDedektoru:
    """
    YOLOv8 tabanli trafik tabelasi ve trafik isigi tespit modulu.
    Egitilmis agirlik dosyasini kullanarak nesne sinifini ve
    sinirlayici kutularini (bounding box) cikarir.
    """

    def __init__(self, model_yolu="modeller/best.pt", guven_esigi=0.45, aktif=True):
        self.guven_esigi = guven_esigi
        self.aktif = aktif
        self.model = None

        if not self.aktif:
            return

        if not os.path.exists(model_yolu):
            print(f"[UYARI] Tabela modeli bulunamadi: {model_yolu}. Tabela tespiti devre disi.")
            self.aktif = False
            return

        try:
            from ultralytics import YOLO
            self.model = YOLO(model_yolu)
            print(f"[BILGI] YOLOv8 modeli yuklendi: {model_yolu}")
        except ImportError:
            print("[UYARI] 'ultralytics' paketi yuklu degil. Tabela tespiti devre disi.")
            self.aktif = False
        except Exception as e:
            print(f"[HATA] Model yuklenirken hata olustu: {e}")
            self.aktif = False

    def tespit_et(self, kare):
        """
        Kamera karesindeki tabelalari tespit eder.
        Dondurulen liste: [{'etiket': str, 'guven': float, 'kutu': (x1, y1, x2, y2)}]
        """
        if not self.aktif or self.model is None:
            return []

        sonuclar = []
        try:
            tahminler = self.model(kare, conf=self.guven_esigi, verbose=False)
            for r in tahminler:
                for box in r.boxes:
                    koordinat = box.xyxy[0].cpu().numpy().astype(int)
                    sinif_id = int(box.cls[0].cpu().numpy())
                    guven = float(box.conf[0].cpu().numpy())
                    etiket = self.model.names.get(sinif_id, f"sinif_{sinif_id}")

                    sonuclar.append({
                        'etiket': etiket,
                        'guven': guven,
                        'kutu': (int(koordinat[0]), int(koordinat[1]), int(koordinat[2]), int(koordinat[3]))
                    })
        except Exception as e:
            print(f"[UYARI] Tabela tespiti sirasinda hata: {e}")

        return sonuclar
