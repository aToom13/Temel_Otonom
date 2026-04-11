import cv2
import numpy as np


class SeritDedektoru:
    """
    Aşama 2: Ters Perspektif Dönüşümü (IPM / Kuş Bakışı) ve Kayan Pencereler (Sliding Windows).
    Görüntüyü kuş bakışı düzleme aktarır, histogram tabanlı kayan pencerelerle
    şerit piksellerini toplar ve 2. dereceden polinom uydurur (x = Ay^2 + By + C).
    Viraj yarıçapını (eğrilik) ve araç merkez sapmasını hesaplar.
    """

    def __init__(self, genislik=640, yukseklik=480, perspektif=None):
        self.genislik = genislik
        self.yukseklik = yukseklik

        # Perspektif dönüşüm noktaları (Varsayılan veya config.yaml)
        if perspektif is None:
            perspektif = {
                'sol_ust': [int(genislik * 0.40), int(yukseklik * 0.58)],
                'sol_alt': [int(genislik * 0.12), int(yukseklik * 0.96)],
                'sag_ust': [int(genislik * 0.60), int(yukseklik * 0.58)],
                'sag_alt': [int(genislik * 0.88), int(yukseklik * 0.96)]
            }

        self.kaynak_noktalar = np.float32([
            perspektif['sol_ust'],
            perspektif['sol_alt'],
            perspektif['sag_ust'],
            perspektif['sag_alt']
        ])

        self.hedef_noktalar = np.float32([
            [int(genislik * 0.20), 0],
            [int(genislik * 0.20), yukseklik],
            [int(genislik * 0.80), 0],
            [int(genislik * 0.80), yukseklik]
        ])

        self.matris = cv2.getPerspectiveTransform(self.kaynak_noktalar, self.hedef_noktalar)
        self.ters_matris = cv2.getPerspectiveTransform(self.hedef_noktalar, self.kaynak_noktalar)

        # Metre - Piksel dönüşüm oranları (Teknofest pist ve kamera kalibrasyonu)
        self.ym_per_pix = 30.0 / 720.0  # y ekseninde metre/piksel
        self.xm_per_pix = 3.7 / 700.0   # x ekseninde metre/piksel

        # Gölge dengeleme filtresi
        self.clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))

        # Geçmiş kareler üzerinde yumuşatma (EMA)
        self.son_sol_fit = None
        self.son_sag_fit = None
        self.son_sapma = 0.0

    def perspektif_cevir(self, kare):
        """Kamera karesini kuş bakışı (IPM) düzleme dönüştürür."""
        return cv2.warpPerspective(kare, self.matris, (self.genislik, self.yukseklik))

    def ters_perspektif_cevir(self, kus_bakisi_kare):
        """Kuş bakışı koordinatları kamera bakış açısına geri yansıtır."""
        return cv2.warpPerspective(kus_bakisi_kare, self.ters_matris, (self.genislik, self.yukseklik))

    def ikili_esikle(self, kare):
        """
        Kuş bakışı veya orijinal görüntüde şerit piksellerini izole eder.
        Gölgeye ve parlak plastik bariyerlere karşı dayanıklıdır.
        """
        hls = cv2.cvtColor(kare, cv2.COLOR_BGR2HLS)
        hsv = cv2.cvtColor(kare, cv2.COLOR_BGR2HSV)
        l_kanali = hls[:, :, 1]
        s_kanali = hls[:, :, 2]

        l_clahe = self.clahe.apply(l_kanali)

        # Beyaz şeritler
        beyaz = cv2.inRange(l_clahe, 185, 255) & cv2.inRange(s_kanali, 0, 85)

        # Sarı şeritler
        sari = cv2.inRange(hsv, np.array([15, 50, 70]), np.array([38, 255, 255]))

        # Plastik kırmızı/turuncu pist bariyerlerini filtrele
        bariyer_1 = cv2.inRange(hsv, np.array([0, 90, 70]), np.array([12, 255, 255]))
        bariyer_2 = cv2.inRange(hsv, np.array([168, 90, 70]), np.array([180, 255, 255]))
        bariyer = cv2.bitwise_or(bariyer_1, bariyer_2)

        renk = cv2.bitwise_or(beyaz, sari)
        renk = cv2.bitwise_and(renk, cv2.bitwise_not(bariyer))

        # Sobel X türevi
        sobelx = cv2.Sobel(l_clahe, cv2.CV_64F, 1, 0, ksize=3)
        abs_sobel = np.absolute(sobelx)
        scaled_sobel = np.uint8(255 * abs_sobel / (np.max(abs_sobel) + 1e-5))
        sobel_maske = cv2.inRange(scaled_sobel, 45, 255)

        birlesik = cv2.bitwise_or(renk, cv2.bitwise_and(sobel_maske, renk | cv2.inRange(l_clahe, 150, 255)))

        cekirdek = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        return cv2.morphologyEx(birlesik, cv2.MORPH_OPEN, cekirdek)

    def kayan_pencereler(self, ikili_kus_bakisi):
        """
        Kuş bakışı ikili görüntü üzerinde dikey kayan pencereler gezdirerek
        sol ve sağ şerit piksellerini toplar ve 2. dereceden polinom uydurur.
        """
        h, w = ikili_kus_bakisi.shape[:2]
        histogram = np.sum(ikili_kus_bakisi[h // 2:, :], axis=0)
        orta_nokta = int(w // 2)

        sol_taban = np.argmax(histogram[:orta_nokta])
        sag_taban = np.argmax(histogram[orta_nokta:]) + orta_nokta

        # Histogram sıfırsa makul taban varsayımları
        if histogram[sol_taban] == 0:
            sol_taban = int(w * 0.20)
        if histogram[sag_taban] == 0:
            sag_taban = int(w * 0.80)

        n_pencere = 10
        pencere_h = h // n_pencere
        kenar_payi = 60
        min_piksel = 30

        nonzero = ikili_kus_bakisi.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        sol_guncel = sol_taban
        sag_guncel = sag_taban

        sol_indisler = []
        sag_indisler = []

        kus_bakisi_cizim = cv2.cvtColor(ikili_kus_bakisi, cv2.COLOR_GRAY2BGR)

        for p in range(n_pencere):
            y_alt = h - (p + 1) * pencere_h
            y_ust = h - p * pencere_h

            x_sol_min = sol_guncel - kenar_payi
            x_sol_max = sol_guncel + kenar_payi
            x_sag_min = sag_guncel - kenar_payi
            x_sag_max = sag_guncel + kenar_payi

            # Kayan pencereleri debug görseline çiz
            cv2.rectangle(kus_bakisi_cizim, (x_sol_min, y_alt), (x_sol_max, y_ust), (0, 255, 120), 1)
            cv2.rectangle(kus_bakisi_cizim, (x_sag_min, y_alt), (x_sag_max, y_ust), (0, 200, 255), 1)

            iyi_sol = ((nonzeroy >= y_alt) & (nonzeroy < y_ust) &
                       (nonzerox >= x_sol_min) & (nonzerox < x_sol_max)).nonzero()[0]
            iyi_sag = ((nonzeroy >= y_alt) & (nonzeroy < y_ust) &
                       (nonzerox >= x_sag_min) & (nonzerox < x_sag_max)).nonzero()[0]

            sol_indisler.append(iyi_sol)
            sag_indisler.append(iyi_sag)

            if len(iyi_sol) > min_piksel:
                sol_guncel = int(np.mean(nonzerox[iyi_sol]))
            if len(iyi_sag) > min_piksel:
                sag_guncel = int(np.mean(nonzerox[iyi_sag]))

        sol_indisler = np.concatenate(sol_indisler) if len(sol_indisler) > 0 else np.array([])
        sag_indisler = np.concatenate(sag_indisler) if len(sag_indisler) > 0 else np.array([])

        solx = nonzerox[sol_indisler] if len(sol_indisler) > 0 else np.array([])
        soly = nonzeroy[sol_indisler] if len(sol_indisler) > 0 else np.array([])
        sagx = nonzerox[sag_indisler] if len(sag_indisler) > 0 else np.array([])
        sagy = nonzeroy[sag_indisler] if len(sag_indisler) > 0 else np.array([])

        ploty = np.linspace(0, h - 1, h)

        # Sol polinom
        if len(solx) > 100:
            sol_fit = np.polyfit(soly, solx, 2)
            self.son_sol_fit = sol_fit
        elif self.son_sol_fit is not None:
            sol_fit = self.son_sol_fit
        else:
            sol_fit = np.array([0.0, 0.0, float(sol_taban)])

        # Sağ polinom
        if len(sagx) > 100:
            sag_fit = np.polyfit(sagy, sagx, 2)
            self.son_sag_fit = sag_fit
        elif self.son_sag_fit is not None:
            sag_fit = self.son_sag_fit
        else:
            sag_fit = np.array([0.0, 0.0, float(sag_taban)])

        sol_fitx = sol_fit[0] * ploty ** 2 + sol_fit[1] * ploty + sol_fit[2]
        sag_fitx = sag_fit[0] * ploty ** 2 + sag_fit[1] * ploty + sag_fit[2]

        # Polinom eğrilerini kuş bakışı görseline çiz
        pts_sol = np.array([np.transpose(np.vstack([sol_fitx, ploty]))], np.int32)
        pts_sag = np.array([np.transpose(np.vstack([sag_fitx, ploty]))], np.int32)
        cv2.polylines(kus_bakisi_cizim, pts_sol, False, (0, 255, 255), 3)
        cv2.polylines(kus_bakisi_cizim, pts_sag, False, (0, 140, 255), 3)

        # Eğrilik ve sapma hesabı
        y_eval = np.max(ploty)
        sol_fit_cr = np.polyfit(ploty * self.ym_per_pix, sol_fitx * self.xm_per_pix, 2)
        sag_fit_cr = np.polyfit(ploty * self.ym_per_pix, sag_fitx * self.xm_per_pix, 2)

        sol_curverad = ((1 + (2 * sol_fit_cr[0] * y_eval * self.ym_per_pix + sol_fit_cr[1]) ** 2) ** 1.5) / (
            abs(2 * sol_fit_cr[0]) + 1e-6
        )
        sag_curverad = ((1 + (2 * sag_fit_cr[0] * y_eval * self.ym_per_pix + sag_fit_cr[1]) ** 2) ** 1.5) / (
            abs(2 * sag_fit_cr[0]) + 1e-6
        )
        egrilik = (sol_curverad + sag_curverad) / 2.0

        yol_merkezi = (sol_fitx[-1] + sag_fitx[-1]) / 2.0
        arac_merkezi = w / 2.0
        sapma_piksel = arac_merkezi - yol_merkezi

        self.son_sapma = 0.80 * self.son_sapma + 0.20 * sapma_piksel

        # Kuş bakışından kamera koordinatlarına doğrusal çizgi eşdeğerleri
        sol_cizgi = (int(sol_fitx[-1]), h, int(sol_fitx[int(h * 0.6)]), int(h * 0.6))
        sag_cizgi = (int(sag_fitx[-1]), h, int(sag_fitx[int(h * 0.6)]), int(h * 0.6))

        return {
            'sol_fit': sol_fit,
            'sag_fit': sag_fit,
            'sol_fitx': sol_fitx,
            'sag_fitx': sag_fitx,
            'ploty': ploty,
            'sol_cizgi': sol_cizgi,
            'sag_cizgi': sag_cizgi,
            'sapma_piksel': self.son_sapma,
            'egrilik_m': egrilik,
            'kus_bakisi_cizim': kus_bakisi_cizim
        }

    def isle(self, kare):
        """Kamera karesini alır, perspektif dönüşümü ve sliding window ile şeritleri çözer."""
        kus_bakisi = self.perspektif_cevir(kare)
        ikili = self.ikili_esikle(kus_bakisi)
        sonuclar = self.kayan_pencereler(ikili)
        return sonuclar['kus_bakisi_cizim'], ikili, sonuclar
