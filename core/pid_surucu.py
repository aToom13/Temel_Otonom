class PIDSurucu:
    """
    Otonom surus yonlendirme ve akilli hiz kontrolcusu.
    Serit merkez sapmasina (offset error) gore PID direksiyon acisini
    hesaplar ve tespit edilen engellere / tabelalara gore hizi dinamik regule eder.
    """

    def __init__(self, kp=0.12, ki=0.01, kd=0.18, max_aci=35.0, taban_hiz=110, min_hiz=70):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.max_aci = max_aci

        self.taban_hiz = taban_hiz
        self.min_hiz = min_hiz

        self.onceki_hata = 0.0
        self.integral = 0.0
        self.integral_sinir = 150.0  # Anti-windup clamping

    def direksiyon_hesapla(self, sapma_piksel, dt=0.033):
        """
        Merkez sapma hatasini PID algoritmasiyla direksiyon acisina cevirir.
        Pozitif sapma: Arac sagda -> Sola kirmali (negatif aci)
        Negatif sapma: Arac solda -> Saga kirmali (pozitif aci)
        """
        hata = -float(sapma_piksel)

        # Integral birikimi ve anti-windup sinirlayici
        self.integral += hata * dt
        self.integral = max(-self.integral_sinir, min(self.integral_sinir, self.integral))

        # Turev hesabi
        turev = (hata - self.onceki_hata) / dt if dt > 0 else 0.0
        self.onceki_hata = hata

        # PID cikisi
        cikis = (self.kp * hata) + (self.ki * self.integral) + (self.kd * turev)

        # Direksiyon donus acisini max_aci ile sinirla
        direksiyon_acisi = max(-self.max_aci, min(self.max_aci, cikis))
        return round(direksiyon_acisi, 1)

    def hiz_belirle(self, direksiyon_acisi, tespitler=None):
        """
        Direksiyon acisi keskinligine ve ondeki aracin/tabelanin mesafesine gore
        guvenli motor hizini (PWM) belirler.
        """
        abs_aci = abs(direksiyon_acisi)

        # Viraj keskinligine gore taban hiz ayari
        if abs_aci > 20.0:
            hiz = self.min_hiz
            durum = "KESKIN VIRAJ"
        elif abs_aci > 8.0:
            hiz = int(self.taban_hiz - (abs_aci * 1.2))
            durum = "VIRAJ"
        else:
            hiz = self.taban_hiz
            durum = "SERIT ORTALAMA (DENGELI)"

        # Nesne ve mesafe duyarliligi
        if tespitler:
            for t in tespitler:
                etiket = t.get('etiket', '').lower()
                mesafe = t.get('mesafe', 999.0)

                # Dur tabelasi veya kirmizi isik
                if any(x in etiket for x in ['dur', 'stop', 'kirmizi', 'red']) and mesafe < 15.0:
                    return 0, f"DUR: {etiket.upper()} ({mesafe:.1f}m)"

                # Onde yakin arac (Takip mesafesi kontrolu)
                if any(x in etiket for x in ['car', 'truck', 'bus', 'arac']) and mesafe < 8.0:
                    hiz = min(hiz, int(self.min_hiz * 0.8))
                    durum = f"YAKIN ARAC ({mesafe:.1f}m)"

        return max(0, hiz), durum

    def hesapla(self, sapma_piksel, tespitler=None, dt=0.033):
        """Direksiyon açısı, hız ve sürüş durumunu tek adımda döndürür."""
        aci = self.direksiyon_hesapla(sapma_piksel, dt=dt)
        hiz, durum = self.hiz_belirle(aci, tespitler)
        return aci, hiz, durum
