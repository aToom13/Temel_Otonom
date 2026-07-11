import cv2
import numpy as np


class Gorsellestirici:
    """
    Otonom sürüş telemetri ve görselleştirme motoru.
    Şerit koridoru, şeridin ortasında fütüristik araç gidiş yolu (trajektori bandı),
    ileri yönlendirme okları (chevrons), hedef nişangah ve cam efektli (Glassmorphism)
    HUD telemetri panelini çizer.
    """

    def __init__(self, genislik=640, yukseklik=480):
        self.genislik = genislik
        self.yukseklik = yukseklik

    def serit_koridoru_ciz(self, kare, serit_bilgisi):
        """
        Sol ve sağ şerit arasına saydam zümrüt yeşili koridoru, şeridin tam ortasına
        parlak araç gidiş yolu (trajektori bandı) ve ileri yönlendirme oklarını (chevrons) çizer.
        """
        sol_cizgi = serit_bilgisi.get('sol_cizgi')
        sag_cizgi = serit_bilgisi.get('sag_cizgi')

        if sol_cizgi is None or sag_cizgi is None:
            return kare

        h, w = kare.shape[:2]
        katman = np.zeros_like(kare)

        sol_alt_x, sol_alt_y, sol_ust_x, sol_ust_y = sol_cizgi
        sag_alt_x, sag_alt_y, sag_ust_x, sag_ust_y = sag_cizgi

        y_min = min(sol_ust_y, sag_ust_y)
        y_max = max(sol_alt_y, sag_alt_y)
        n_pts = 30
        y_vals = np.linspace(y_min, y_max, n_pts)

        # Polinom veya doğrusal x koordinatları
        sol_x = np.linspace(sol_ust_x, sol_alt_x, n_pts)
        sag_x = np.linspace(sag_ust_x, sag_alt_x, n_pts)
        orta_x = (sol_x + sag_x) / 2.0

        # 1. ŞERİT KORİDORU (Zümrüt Yeşili Şeffaf Poligon)
        pts_sol = np.column_stack((sol_x, y_vals))
        pts_sag = np.column_stack((sag_x, y_vals))[::-1]
        koridor_poligon = np.vstack([pts_sol, pts_sag]).astype(np.int32)
        cv2.fillPoly(katman, [koridor_poligon], (0, 205, 100))

        # Şerit sınır çizgileri (Sol: Cyan/Mavi, Sağ: Amber/Sarı)
        cv2.polylines(katman, [pts_sol.astype(np.int32)], False, (255, 225, 0), 5, cv2.LINE_AA)
        cv2.polylines(katman, [pts_sag.astype(np.int32)], False, (0, 165, 255), 5, cv2.LINE_AA)

        # 2. ARAÇ GİDİŞ YOLU (ORTA TRAJEKTORİ BANDI)
        # Yolun tabanından ufka doğru daralan dinamik sürüş izi
        genislik_alt = 46.0   # Araç önü gidiş yolu genişliği
        genislik_ust = 16.0   # Ufuk noktası gidiş yolu genişliği
        t = np.linspace(0.0, 1.0, n_pts)  # 0: ufuk, 1: kaput
        yol_yarim_genislik = (genislik_ust + t * (genislik_alt - genislik_ust)) / 2.0

        yol_sol_x = orta_x - yol_yarim_genislik
        yol_sag_x = orta_x + yol_yarim_genislik

        pts_yol_sol = np.column_stack((yol_sol_x, y_vals))
        pts_yol_sag = np.column_stack((yol_sag_x, y_vals))[::-1]
        gidis_yolu_poligon = np.vstack([pts_yol_sol, pts_yol_sag]).astype(np.int32)

        # Dış halesi: Parlak Elektrik Cyan (Neon Glow)
        cv2.fillPoly(katman, [gidis_yolu_poligon], (255, 235, 40))

        # Çekirdek çizgisi: Parlak Beyaz/Camgöbeği
        pts_orta = np.column_stack((orta_x, y_vals)).astype(np.int32)
        cv2.polylines(katman, [pts_orta], False, (255, 255, 240), 3, cv2.LINE_AA)

        # 3. İLERİ YÖNLENDİRME OKLARI (CHEVRONS)
        # Araç gidiş yolu üzerinde ufka doğru sıralanmış kılavuz oklar
        chevron_indisleri = [5, 11, 17, 23]
        for idx in chevron_indisleri:
            cx = int(orta_x[idx])
            cy = int(y_vals[idx])
            olcek = 0.5 + 0.5 * t[idx]
            kanat_w = int(14 * olcek)
            kanat_h = int(10 * olcek)

            pt_orta = (cx, cy - kanat_h)
            pt_sol = (cx - kanat_w, cy + kanat_h)
            pt_sag = (cx + kanat_w, cy + kanat_h)

            cv2.line(katman, pt_sol, pt_orta, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.line(katman, pt_sag, pt_orta, (255, 255, 255), 2, cv2.LINE_AA)

        # 4. HEDEF NİŞANGAH (LOOKAHEAD TARGET WAYPOINT)
        hedef_idx = 4
        hx = int(orta_x[hedef_idx])
        hy = int(y_vals[hedef_idx])
        cv2.circle(katman, (hx, hy), 7, (0, 245, 255), 2, cv2.LINE_AA)
        cv2.circle(katman, (hx, hy), 2, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.line(katman, (hx - 12, hy), (hx + 12, hy), (0, 245, 255), 1, cv2.LINE_AA)
        cv2.line(katman, (hx, hy - 12), (hx, hy + 12), (0, 245, 255), 1, cv2.LINE_AA)

        # Katmanı ana kare ile harmanla
        cikti = cv2.addWeighted(kare, 1.0, katman, 0.42, 0)
        return cikti

    def telemetri_paneli_ciz(self, kare, aci, hiz, durum, sapma, fps, nesne_bilgisi=None, egrilik=None):
        """Sol üst köşeye modern ve şık bir Glassmorphism HUD paneli yerleştirir."""
        kare_cikti = kare.copy()
        card_x, card_y, card_w, card_h = 20, 18, 310, 205

        # Yarı saydam koyu zemin
        overlay = kare_cikti.copy()
        cv2.rectangle(overlay, (card_x, card_y), (card_x + card_w, card_y + card_h), (12, 16, 24), -1)
        cv2.addWeighted(overlay, 0.86, kare_cikti, 0.14, 0, kare_cikti)
        cv2.rectangle(kare_cikti, (card_x, card_y), (card_x + card_w, card_y + card_h), (40, 55, 75), 1)

        # Başlık rozeti
        cv2.rectangle(kare_cikti, (card_x + 10, card_y + 10), (card_x + card_w - 10, card_y + 38), (22, 30, 42), -1)
        cv2.circle(kare_cikti, (card_x + 24, card_y + 24), 4, (0, 255, 130), -1)
        cv2.putText(kare_cikti, 'TEMEL OTONOM SURUS v2.4', (card_x + 36, card_y + 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (230, 242, 255), 1, cv2.LINE_AA)
        cv2.putText(kare_cikti, f'{fps:.1f} FPS', (card_x + card_w - 75, card_y + 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 255, 130), 1, cv2.LINE_AA)

        # Yön bilgisi metni
        yon_str = "DUZ" if abs(aci) < 2.0 else ("SOLA" if aci < 0 else "SAGA")

        satirlar = [
            ('Direksiyon Acisi:', f'{aci:+.1f} deg ({yon_str})', (0, 235, 255)),
            ('Serit Sapmasi:', f'{sapma:+.1f} px ({sapma*0.5:+.1f} cm)',
             (0, 255, 140) if abs(sapma) < 25 else (255, 180, 50)),
            ('Seyir Hizi (PWM):', f'{hiz}', (225, 235, 245)),
            ('Surus Durumu:', durum, (0, 255, 130) if 'DENGELI' in durum or 'DUZ' in durum else (255, 180, 50))
        ]

        if nesne_bilgisi:
            satirlar.insert(3, ('Algilanan Nesne:', nesne_bilgisi, (0, 215, 255)))

        if egrilik is not None and egrilik > 0 and egrilik < 5000:
            satirlar.append(('Viraj Yaricapi:', f'{egrilik:.0f} m', (200, 220, 255)))

        y_konum = card_y + 60
        for etiket, deger, renk in satirlar:
            cv2.putText(kare_cikti, etiket, (card_x + 14, y_konum),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.37, (160, 175, 190), 1, cv2.LINE_AA)
            cv2.putText(kare_cikti, str(deger), (card_x + 145, y_konum),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.39, renk, 1, cv2.LINE_AA)
            y_konum += 22

        # Direksiyon mini göstergesi
        gx = card_x + card_w // 2
        gy = card_y + card_h - 18
        cv2.line(kare_cikti, (gx - 60, gy), (gx + 60, gy), (65, 80, 100), 2)
        cv2.circle(kare_cikti, (gx, gy), 3, (140, 160, 180), -1)

        sapma_pixel = int(np.clip(aci * 1.6, -55, 55))
        cv2.circle(kare_cikti, (gx + sapma_pixel, gy), 5, (0, 235, 255), -1)
        cv2.putText(kare_cikti, 'L', (gx - 74, gy + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (120, 135, 150), 1, cv2.LINE_AA)
        cv2.putText(kare_cikti, 'R', (gx + 66, gy + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (120, 135, 150), 1, cv2.LINE_AA)

        return kare_cikti

    def coklu_panel_olustur(self, ana_kare, kus_bakisi, maske):
        """Geliştirici modu: Ana görüntü, kuş bakışı ve ikili şerit maskesini birleştirir."""
        h, w = ana_kare.shape[:2]
        sub_h = h // 2
        sub_w = int(sub_h * (4 / 3))

        sub1 = cv2.resize(kus_bakisi, (sub_w, sub_h))
        if len(maske.shape) == 2:
            maske_renkli = cv2.cvtColor(maske, cv2.COLOR_GRAY2BGR)
        else:
            maske_renkli = maske
        sub2 = cv2.resize(maske_renkli, (sub_w, sub_h))

        # Panel başlık şeritleri
        cv2.rectangle(sub1, (0, 0), (sub_w, 24), (16, 22, 30), -1)
        cv2.putText(sub1, 'KUS BAKISI / PERSPEKTIF (IPM)', (8, 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36, (0, 230, 255), 1, cv2.LINE_AA)

        cv2.rectangle(sub2, (0, 0), (sub_w, 24), (16, 22, 30), -1)
        cv2.putText(sub2, 'IKILI SERIT MASKESI', (8, 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36, (0, 255, 130), 1, cv2.LINE_AA)

        sag_kolon = np.vstack([sub1, sub2])
        ana_olcekli = cv2.resize(ana_kare, (int(h * (16 / 9)), h))
        return np.hstack([ana_olcekli, sag_kolon])
