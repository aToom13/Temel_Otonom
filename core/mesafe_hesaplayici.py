class MesafeHesaplayici:
    """
    Kamera odak uzakligi (focal length) ve bilinen nesne gercek genisligi
    kullanilarak bounding box boyutundan gercek dunya mesafesi (metre) kestirimi yapar.
    Formul: Mesafe = (Gercek_Genislik * Odak_Uzakligi) / BoundingBox_Genisligi
    """

    def __init__(self, odak_uzakligi=950.0):
        self.odak_uzakligi = odak_uzakligi

        # Bilinen gercek dunya genislikleri (metre cinsinden)
        self.bilinen_genislikler = {
            'car': 1.85,
            'truck': 2.50,
            'bus': 2.60,
            'motorcycle': 0.85,
            'person': 0.50,
            'traffic_sign': 0.60,
            'tabela': 0.60,
            'isik': 0.35,
            'varsayilan': 1.80
        }

    def hesapla(self, etiket, kutu_genisligi_px):
        """Tespit edilen nesnenin araca olan mesafesini metre cinsinden kestirir."""
        if kutu_genisligi_px <= 0:
            return 999.0

        gercek_genislik = self.bilinen_genislikler.get(etiket, self.bilinen_genislikler['varsayilan'])
        mesafe = (gercek_genislik * self.odak_uzakligi) / float(kutu_genisligi_px)
        return round(mesafe, 2)
