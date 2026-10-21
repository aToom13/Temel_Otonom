import os
import cv2
import numpy as np


class YOLOSeritDedektoru:
    """
    Derin Öğrenme (YOLOv8 Lane Segmentation) Tabanlı Şerit Takip Motoru.

    Geliştirme Amacı (Güneş - Gölge Problemi Çözümü):
    Geleneksel renk filtreleri (HSV/HLS thresholding), güneşli günlerde yarışma
    parkurlarındaki çadır direkleri, ağaçlar veya köprülerin oluşturduğu sert
    gölgelerde piksel parlaklığının (V kanalı) aniden düşmesi nedeniyle şerit
    çizgilerini kaybeder veya güneş altındaki açık renkli asfaltı/bariyerleri
    şerit zannederek hataya düşer.

    YOLOv8 tabanlı semantik segmentasyon (Lane Segmentation) modeli; piksel bazlı
    statik renk eşikleri yerine, konvolüsyonel katmanlar aracılığıyla şeridin
    biçimini, sürekliliğini ve yol geometrisini öğrendiği için derin gölgeler
    altında dahi kesintisiz ve kararlı şerit maskesi üretir.
    """

    def __init__(self, model_yolu="modeller/yolov8_serit_seg.pt", guven_esigi=0.30, genislik=640, yukseklik=480):
        self.model_yolu = model_yolu
        self.guven_esigi = guven_esigi
        self.genislik = genislik
        self.yukseklik = yukseklik
        self.model = None
        self.yolo_aktif = False

        self.son_sol_cizgi = None
        self.son_sag_cizgi = None
        self.son_sapma = 0.0

        self._model_yukle()

    def _model_yukle(self):
        """Ultralytics YOLO segmentasyon modelini yükler; yoksa gölgeye dayanıklı adaptif moda geçer."""
        try:
            from ultralytics import YOLO
            if os.path.exists(self.model_yolu):
                self.model = YOLO(self.model_yolu)
                self.yolo_aktif = True
                print(f"[BILGI] YOLOv8 Şerit Segmentasyon modeli yüklendi: {self.model_yolu}")
        except Exception as e:
            print(f"[BILGI] YOLOv8 çalışma ortamı ({e}); gölge dayanımlı adaptif derin filtreleme devrede.")

    def golge_dayanimli_on_isleme(self, kare):
        """
        YOLO modeli aktif olmadığında veya gölge bölgelerinde HSV'nin çökmesini
        önleyen CLAHE + LAB parlaklık eşitleme ve adaptif eşikleme filtresi.
        """
        h, w = kare.shape[:2]
        lab = cv2.cvtColor(kare, cv2.COLOR_BGR2LAB)
        l_chan, a_chan, b_chan = cv2.split(lab)

        # Kontrast Sınırlı Adaptif Histogram Eşitleme (Sert gölgeleri açar)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l_norm = clahe.apply(l_chan)
        lab_norm = cv2.merge([l_norm, a_chan, b_chan])
        bgr_norm = cv2.cvtColor(lab_norm, cv2.COLOR_LAB2BGR)

        # Gri ton ve adaptif ikili maske
        gri = cv2.cvtColor(bgr_norm, cv2.COLOR_BGR2GRAY)
        blurlu = cv2.GaussianBlur(gri, (5, 5), 0)
        adaptif_mask = cv2.adaptiveThreshold(
            blurlu, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 19, -4
        )

        # ROI yamuk maskesi
        maske_roi = np.zeros_like(adaptif_mask)
        pts = np.array([[
            (int(w * 0.05), int(h * 0.95)),
            (int(w * 0.35), int(h * 0.55)),
            (int(w * 0.65), int(h * 0.55)),
            (int(w * 0.95), int(h * 0.95))
        ]], dtype=np.int32)
        cv2.fillPoly(maske_roi, pts, 255)

        return cv2.bitwise_and(adaptif_mask, maske_roi)

    def maskeden_seritleri_cikar(self, binary_maske, kare_w, kare_h):
        """İkili şerit maskesinden sol ve sağ şerit koordinatlarını ve merkez sapmasını türetir."""
        y_min = int(kare_h * 0.58)
        y_max = int(kare_h * 0.92)

        # Sol ve sağ yarı pikselleri ayıkla
        mid_x = kare_w // 2
        nonzero = binary_maske.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        sol_inds = (nonzerox < mid_x) & (nonzeroy >= y_min) & (nonzeroy <= y_max)
        sag_inds = (nonzerox >= mid_x) & (nonzeroy >= y_min) & (nonzeroy <= y_max)

        sol_x = nonzerox[sol_inds]
        sol_y = nonzeroy[sol_inds]
        sag_x = nonzerox[sag_inds]
        sag_y = nonzeroy[sag_inds]

        # Sol şerit uydurma
        if len(sol_x) > 50:
            sol_fit = np.polyfit(sol_y, sol_x, 1)
            sol_alt = int(sol_fit[0] * y_max + sol_fit[1])
            sol_ust = int(sol_fit[0] * y_min + sol_fit[1])
            sol_cizgi = (sol_alt, y_max, sol_ust, y_min)
            self.son_sol_cizgi = sol_cizgi
        elif self.son_sol_cizgi is not None:
            sol_cizgi = self.son_sol_cizgi
        else:
            sol_cizgi = (int(kare_w * 0.18), y_max, int(kare_w * 0.44), y_min)

        # Sağ şerit uydurma
        if len(sag_x) > 50:
            sag_fit = np.polyfit(sag_y, sag_x, 1)
            sag_alt = int(sag_fit[0] * y_max + sag_fit[1])
            sag_ust = int(sag_fit[0] * y_min + sag_fit[1])
            sag_cizgi = (sag_alt, y_max, sag_ust, y_min)
            self.son_sag_cizgi = sag_cizgi
        elif self.son_sag_cizgi is not None:
            sag_cizgi = self.son_sag_cizgi
        else:
            sag_cizgi = (int(kare_w * 0.82), y_max, int(kare_w * 0.56), y_min)

        # Yol merkezinden sapma (Lateral Offset)
        yol_merkezi = (sol_cizgi[0] + sag_cizgi[0]) / 2.0
        arac_merkezi = kare_w / 2.0
        sapma = arac_merkezi - yol_merkezi

        self.son_sapma = 0.80 * self.son_sapma + 0.20 * sapma
        return sol_cizgi, sag_cizgi, self.son_sapma

    def tespit_et(self, kare):
        """Kamera karesini işleyip YOLO segmentasyon maskesini ve şerit bilgilerini döndürür."""
        h, w = kare.shape[:2]
        binary_maske = None

        if self.model is not None:
            try:
                sonuclar = self.model.predict(source=kare, conf=self.guven_esigi, verbose=False)
                for r in sonuclar:
                    if r.masks is not None and len(r.masks.data) > 0:
                        masks = r.masks.data.cpu().numpy()
                        combined = np.any(masks, axis=0).astype(np.uint8) * 255
                        binary_maske = cv2.resize(combined, (w, h), interpolation=cv2.INTER_NEAREST)
                        break
            except Exception:
                binary_maske = None

        if binary_maske is None:
            binary_maske = self.golge_dayanimli_on_isleme(kare)

        sol_cizgi, sag_cizgi, sapma = self.maskeden_seritleri_cikar(binary_maske, w, h)

        return {
            'sol_cizgi': sol_cizgi,
            'sag_cizgi': sag_cizgi,
            'sapma_piksel': sapma,
            'maske': binary_maske,
            'yolo_aktif': self.yolo_aktif
        }
