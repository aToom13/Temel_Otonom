import cv2
import numpy as np


class KlasikSeritDedektoru:
    """
    Aşama 1: Geleneksel Görüntü İşleme Tabanlı Şerit Tespiti.
    CLAHE gölge dengeleme, HLS/HSV renk filtreleme, plastik bariyer bastırma,
    ilgi alanı (ROI) maskeleme, Probabilistik Hough Çizgi Dönüşümü (HoughLinesP)
    ve doğrusal polinom uydurma (Linear Fit) ile şeritleri ve araç gidiş yolunu çıkarır.
    """

    def __init__(self, genislik=640, yukseklik=480):
        self.genislik = genislik
        self.yukseklik = yukseklik

        # Gölge dengeleme için CLAHE filtresi
        self.clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))

        # Geçmiş kareler üzerinde yumuşatma (EMA)
        self.son_sol_cizgi = None
        self.son_sag_cizgi = None
        self.son_sapma = 0.0

    def on_isle(self, kare):
        """
        Gölgeye ve plastik bariyerlere karşı dayanıklı ikili şerit maskesi üretir.
        Teknofest ve pist testlerinde bariyerlerin (kırmızı/turuncu plastik) yanlış
        şerit olarak algılanmasını önler.
        """
        hls = cv2.cvtColor(kare, cv2.COLOR_BGR2HLS)
        hsv = cv2.cvtColor(kare, cv2.COLOR_BGR2HSV)
        l_kanali = hls[:, :, 1]
        s_kanali = hls[:, :, 2]

        # 1. CLAHE ile gölgeli alanların parlaklığını normalize et
        l_clahe = self.clahe.apply(l_kanali)

        # 2. Beyaz şerit maskesi (Yüksek parlaklık, düşük doygunluk)
        beyaz_maske = cv2.inRange(l_clahe, 190, 255) & cv2.inRange(s_kanali, 0, 90)

        # 3. Sarı şerit maskesi (HSV: Hue 15-38, Doygunluk > 50)
        sari_maske = cv2.inRange(hsv, np.array([15, 50, 70]), np.array([38, 255, 255]))

        # 4. Plastik pist bariyerlerini bastırma (Kırmızı/Turuncu doygun renkler)
        bariyer_1 = cv2.inRange(hsv, np.array([0, 90, 70]), np.array([12, 255, 255]))
        bariyer_2 = cv2.inRange(hsv, np.array([168, 90, 70]), np.array([180, 255, 255]))
        bariyer_maskesi = cv2.bitwise_or(bariyer_1, bariyer_2)

        # Şerit renklerini birleştir ve bariyerleri temizle
        renk_maskesi = cv2.bitwise_or(beyaz_maske, sari_maske)
        renk_maskesi = cv2.bitwise_and(renk_maskesi, cv2.bitwise_not(bariyer_maskesi))

        # 5. Kenar tespiti (Canny)
        blurlu = cv2.GaussianBlur(l_clahe, (5, 5), 0)
        kenarlar = cv2.Canny(blurlu, 50, 150)

        # Renk maskesi ile kenar bilgilerini harmanla
        birlesik = cv2.bitwise_or(kenarlar, renk_maskesi)

        # Morfolojik gürültü temizleme
        cekirdek = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        temiz = cv2.morphologyEx(birlesik, cv2.MORPH_OPEN, cekirdek)
        return temiz

    def roi_maskele(self, kare):
        """Kamera açısına göre yol harici alanları (gökyüzü, kaput, ufuk) maskeler."""
        h, w = kare.shape[:2]
        poligon = np.array([[
            (int(w * 0.05), h),
            (int(w * 0.40), int(h * 0.58)),
            (int(w * 0.60), int(h * 0.58)),
            (int(w * 0.95), h)
        ]], dtype=np.int32)

        maske = np.zeros_like(kare)
        cv2.fillPoly(maske, poligon, 255)
        return cv2.bitwise_and(kare, maske)

    def cizgileri_ayikla(self, maskeli_kenar):
        """
        Hough dönüşümü ile çizgileri yakalayıp eğimlerine ve konumlarına
        göre sol ve sağ şerit olarak ayırır.
        """
        h, w = maskeli_kenar.shape[:2]
        cizgiler = cv2.HoughLinesP(
            maskeli_kenar,
            rho=2,
            theta=np.pi / 180,
            threshold=40,
            lines=np.array([]),
            minLineLength=30,
            maxLineGap=35
        )

        sol_x, sol_y = [], []
        sag_x, sag_y = [], []

        if cizgiler is not None:
            for cizgi in cizgiler:
                # OpenCV 4 ve 5 arasındaki şekil farkına karşı evrensel düzleştirme
                x1, y1, x2, y2 = [int(v) for v in cizgi.flatten()[:4]]
                dx = x2 - x1
                if dx == 0:
                    continue
                egim = (y2 - y1) / float(dx)

                # Yatay ve dik açı anomalilerini filtrele
                if abs(egim) < 0.40 or abs(egim) > 3.2:
                    continue

                # Sol şerit (negatif eğim, yolun sol yarı alanı)
                if egim < 0 and x1 < w * 0.55 and x2 < w * 0.55:
                    sol_x.extend([x1, x2])
                    sol_y.extend([y1, y2])
                # Sağ şerit (pozitif eğim, yolun sağ yarı alanı)
                elif egim > 0 and x1 > w * 0.45 and x2 > w * 0.45:
                    sag_x.extend([x1, x2])
                    sag_y.extend([y1, y2])

        min_y = int(h * 0.60)
        max_y = h

        # Sol şerit doğrusu uydurma
        if len(sol_x) >= 2:
            sol_fit = np.polyfit(sol_y, sol_x, 1)
            sol_alt = int(sol_fit[0] * max_y + sol_fit[1])
            sol_ust = int(sol_fit[0] * min_y + sol_fit[1])
            sol_cizgi = (sol_alt, max_y, sol_ust, min_y)
            self.son_sol_cizgi = sol_cizgi
        elif self.son_sol_cizgi is not None:
            sol_cizgi = self.son_sol_cizgi
        else:
            sol_cizgi = (int(w * 0.18), max_y, int(w * 0.45), min_y)

        # Sağ şerit doğrusu uydurma
        if len(sag_x) >= 2:
            sag_fit = np.polyfit(sag_y, sag_x, 1)
            sag_alt = int(sag_fit[0] * max_y + sag_fit[1])
            sag_ust = int(sag_fit[0] * min_y + sag_fit[1])
            sag_cizgi = (sag_alt, max_y, sag_ust, min_y)
            self.son_sag_cizgi = sag_cizgi
        elif self.son_sag_cizgi is not None:
            sag_cizgi = self.son_sag_cizgi
        else:
            sag_cizgi = (int(w * 0.82), max_y, int(w * 0.55), min_y)

        # Araç merkezinden yol merkezine sapma hesabı
        yol_merkezi = (sol_cizgi[0] + sag_cizgi[0]) / 2.0
        arac_merkezi = w / 2.0
        sapma = arac_merkezi - yol_merkezi

        # EMA filtresi ile sarsıntıyı önle
        self.son_sapma = 0.80 * self.son_sapma + 0.20 * sapma

        # Şerit ortasındaki araç gidiş yolu güzergah koordinatları
        y_noktalari = np.linspace(min_y, max_y, 25).astype(int)
        sol_x_arr = np.linspace(sol_cizgi[2], sol_cizgi[0], 25)
        sag_x_arr = np.linspace(sag_cizgi[2], sag_cizgi[0], 25)
        orta_x_arr = (sol_x_arr + sag_x_arr) / 2.0
        merkez_noktalar = list(zip(orta_x_arr.astype(int), y_noktalari))

        return sol_cizgi, sag_cizgi, self.son_sapma, merkez_noktalar

    def tespit_et(self, kare):
        """Kamera karesini işleyip şerit çizgilerini, gidiş yolunu ve sapmayı döndürür."""
        ikili = self.on_isle(kare)
        maskeli = self.roi_maskele(ikili)
        sol_cizgi, sag_cizgi, sapma, merkez_noktalar = self.cizgileri_ayikla(maskeli)

        return {
            'sol_cizgi': sol_cizgi,
            'sag_cizgi': sag_cizgi,
            'sapma_piksel': sapma,
            'merkez_noktalar': merkez_noktalar,
            'maskeli_kenar': maskeli,
            'ikili_maske': ikili
        }
