import unittest
import numpy as np
import cv2
import os

from core.klasik_serit import KlasikSeritDedektoru
from core.serit_tespit import SeritDedektoru
from core.mesafe_hesaplayici import MesafeHesaplayici
from core.yolo_tespit import YOLODedektoru
from core.yolo_serit import YOLOSeritDedektoru
from core.pid_surucu import PIDSurucu
from araclar.cizici import Gorsellestirici
from donanim.arduino_haberlesme import ArduinoBaglanti


class TestKlasikSeritDedektoru(unittest.TestCase):
    def setUp(self):
        self.dedektor = KlasikSeritDedektoru(640, 480)

    def test_bos_kare(self):
        """Tamamen siyah karede algoritma çökmeden güvenli varsayılan değer üretmeli."""
        kare = np.zeros((480, 640, 3), dtype=np.uint8)
        sonuc = self.dedektor.tespit_et(kare)
        self.assertIn('sol_cizgi', sonuc)
        self.assertIn('sag_cizgi', sonuc)
        self.assertIn('sapma_piksel', sonuc)
        self.assertIn('merkez_noktalar', sonuc)
        self.assertEqual(len(sonuc['sol_cizgi']), 4)
        self.assertEqual(len(sonuc['sag_cizgi']), 4)

    def test_beyaz_kare(self):
        """Tamamen beyaz karede algoritma çökmeden güvenli varsayılan değer üretmeli."""
        kare = np.full((480, 640, 3), 255, dtype=np.uint8)
        sonuc = self.dedektor.tespit_et(kare)
        self.assertIsNotNone(sonuc['sol_cizgi'])
        self.assertIsNotNone(sonuc['sag_cizgi'])

    def test_sentetik_seritli_kare(self):
        """Sol ve sağ şerit çizilmiş sentetik karede sapmanın merkeze yakın çıkması."""
        kare = np.zeros((480, 640, 3), dtype=np.uint8)
        # Sol beyaz şerit
        cv2.line(kare, (140, 480), (280, 280), (255, 255, 255), 8)
        # Sağ sarı şerit
        cv2.line(kare, (500, 480), (360, 280), (0, 255, 255), 8)
        sonuc = self.dedektor.tespit_et(kare)
        self.assertAlmostEqual(sonuc['sapma_piksel'], 0.0, delta=40.0)


class TestSeritDedektoruKusBakisi(unittest.TestCase):
    def setUp(self):
        self.dedektor = SeritDedektoru(640, 480)

    def test_perspektif_donusum(self):
        """Perspektif ve ters perspektif dönüşümü boyutları korumalı."""
        kare = np.zeros((480, 640, 3), dtype=np.uint8)
        kb = self.dedektor.perspektif_cevir(kare)
        ters = self.dedektor.ters_perspektif_cevir(kb)
        self.assertEqual(kb.shape, (480, 640, 3))
        self.assertEqual(ters.shape, (480, 640, 3))

    def test_kayan_pencereler_bos_kare(self):
        """Piksel bulunmayan ikili görüntüde sistem varsayılan polinom üretmeli."""
        ikili = np.zeros((480, 640), dtype=np.uint8)
        sonuc = self.dedektor.kayan_pencereler(ikili)
        self.assertIn('sol_fit', sonuc)
        self.assertIn('sag_fit', sonuc)
        self.assertIn('egrilik_m', sonuc)
        self.assertGreater(sonuc['egrilik_m'], 0)


class TestPIDSurucu(unittest.TestCase):
    def setUp(self):
        self.pid = PIDSurucu(kp=0.15, ki=0.01, kd=0.10, max_aci=35.0, taban_hiz=110, min_hiz=70)

    def test_sifir_hata(self):
        """Sıfır merkez hatasında direksiyon açısı 0 olmalı."""
        aci = self.pid.direksiyon_hesapla(0.0)
        self.assertEqual(aci, 0.0)

    def test_pozitif_hata_sola_yonelme(self):
        """Pozitif piksel sapmasında (araç sağda) negatif direksiyon açısı (sola) üretilmeli."""
        aci = self.pid.direksiyon_hesapla(50.0)
        self.assertLess(aci, 0.0)

    def test_negatif_hata_saga_yonelme(self):
        """Negatif piksel sapmasında (araç solda) pozitif direksiyon açısı (sağa) üretilmeli."""
        aci = self.pid.direksiyon_hesapla(-50.0)
        self.assertGreater(aci, 0.0)

    def test_max_aci_sinirlama(self):
        """Aşırı büyük hata durumunda max_aci sınırı aşılmamalı."""
        aci = self.pid.direksiyon_hesapla(1000.0)
        self.assertEqual(aci, -35.0)

    def test_engel_ve_dur_tabelasi(self):
        """Yakın DUR tabelası veya kırmızı ışıkta hız 0 olmalı."""
        tespitler = [{'etiket': 'dur', 'mesafe': 5.0, 'kutu': (0, 0, 10, 10)}]
        hiz, durum = self.pid.hiz_belirle(0.0, tespitler)
        self.assertEqual(hiz, 0)
        self.assertIn("DUR", durum)

    def test_viraj_hiz_kisitlamasi(self):
        """Yüksek direksiyon açısında hız min_hiz seviyesine düşürülmeli."""
        hiz, durum = self.pid.hiz_belirle(25.0)
        self.assertEqual(hiz, 70)
        self.assertIn("VIRAJ", durum)


class TestMesafeHesaplayici(unittest.TestCase):
    def setUp(self):
        self.hesaplayici = MesafeHesaplayici(odak_uzakligi=1000.0)

    def test_mesafe_dogrulugu(self):
        """Formül: Mesafe = (Gerçek_Genişlik * f) / w_bbox"""
        # Araç genişliği 1.85m, f=1000, bbox=185px -> Mesafe = (1.85 * 1000) / 185 = 10.0m
        mesafe = self.hesaplayici.hesapla('car', 185)
        self.assertAlmostEqual(mesafe, 10.0, places=1)

    def test_gecersiz_kutu_genisligi(self):
        """Sıfır veya negatif genişlikte güvenli 999.0m dönmeli."""
        self.assertEqual(self.hesaplayici.hesapla('car', 0), 999.0)
        self.assertEqual(self.hesaplayici.hesapla('car', -10), 999.0)


class TestArduinoHaberlesme(unittest.TestCase):
    def test_ascii_ha_protokol(self):
        """ASCII H<hiz>A<aci> formatı doğrulaması."""
        arduino = ArduinoBaglanti(simulasyon=True, protokol="ascii_ha")
        paket = arduino.komut_olustur(hiz=110, aci=-12.5)
        self.assertEqual(paket, "H110A-12.5\n")

    def test_kat4_rv_protokol(self):
        """Kat4 Teknofest Robotaksi formatı doğrulaması (r/l/f + aci + v + hiz)."""
        arduino = ArduinoBaglanti(simulasyon=True, protokol="kat4_rv")
        # Sağa dönüş
        self.assertEqual(arduino.komut_olustur(hiz=110, aci=15.0), "r015v110\n")
        # Sola dönüş
        self.assertEqual(arduino.komut_olustur(hiz=95, aci=-20.0), "l020v095\n")
        # Düz gidiş
        self.assertEqual(arduino.komut_olustur(hiz=120, aci=0.0), "f000v120\n")


class TestGorsellestirici(unittest.TestCase):
    def setUp(self):
        self.gorsel = Gorsellestirici(640, 480)

    def test_serit_koridoru_ve_gidis_yolu(self):
        """Şerit koridoru ve araç gidiş yolu çizimi boyutu değiştirmemeli."""
        kare = np.zeros((480, 640, 3), dtype=np.uint8)
        serit_bilgisi = {
            'sol_cizgi': (120, 480, 260, 280),
            'sag_cizgi': (520, 480, 380, 280)
        }
        cizilmis = self.gorsel.serit_koridoru_ciz(kare, serit_bilgisi)
        self.assertEqual(cizilmis.shape, (480, 640, 3))
        # Çizilen pikseller siyah kalmamalı
        self.assertGreater(np.sum(cizilmis), 0)

    def test_eksik_cizgi_guvenligi(self):
        """Şerit çizgisi None ise hata vermeden orijinal kare dönmeli."""
        kare = np.zeros((480, 640, 3), dtype=np.uint8)
        cizilmis = self.gorsel.serit_koridoru_ciz(kare, {'sol_cizgi': None})
        self.assertTrue(np.array_equal(kare, cizilmis))

    def test_coklu_panel_olustur(self):
        """Çoklu panel geniş ekran düzenini doğrula."""
        ana = np.zeros((480, 640, 3), dtype=np.uint8)
        kb = np.zeros((480, 640, 3), dtype=np.uint8)
        mask = np.zeros((480, 640), dtype=np.uint8)
        panel = self.gorsel.coklu_panel_olustur(ana, kb, mask)
        self.assertEqual(panel.shape[0], 480)
        self.assertGreater(panel.shape[1], 640)


class TestVideoPipelineEntegrasyon(unittest.TestCase):
    def test_otoban_video_akisi(self):
        """otoban.mp4 videosundan 5 kare işleme testi."""
        cap = cv2.VideoCapture('test_verileri/otoban.mp4')
        self.assertTrue(cap.isOpened(), "otoban.mp4 açılamadı")
        dedektor = KlasikSeritDedektoru(640, 480)
        for _ in range(5):
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, (640, 480))
            sonuc = dedektor.tespit_et(frame)
            self.assertIn('sol_cizgi', sonuc)
            self.assertIn('sag_cizgi', sonuc)
        cap.release()

    def test_golge_video_akisi(self):
        """golge.webm videosundan 5 kare işleme testi."""
        cap = cv2.VideoCapture('test_verileri/golge.webm')
        self.assertTrue(cap.isOpened(), "golge.webm açılamadı")
        dedektor = KlasikSeritDedektoru(640, 480)
        for _ in range(5):
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, (640, 480))
            sonuc = dedektor.tespit_et(frame)
            self.assertIn('sol_cizgi', sonuc)
        cap.release()


class TestYOLOSeritDedektoru(unittest.TestCase):
    def setUp(self):
        self.dedektor = YOLOSeritDedektoru(genislik=640, yukseklik=480)

    def test_bos_kare_guvenligi(self):
        """Siyah karede algoritma çökmeden geçerli şerit koordinatları üretmeli."""
        kare = np.zeros((480, 640, 3), dtype=np.uint8)
        sonuc = self.dedektor.tespit_et(kare)
        self.assertIn('sol_cizgi', sonuc)
        self.assertIn('sag_cizgi', sonuc)
        self.assertIn('sapma_piksel', sonuc)
        self.assertIn('maske', sonuc)

    def test_golge_video_akisi(self):
        """golge.webm videosundan kareler üzerinde YOLO şerit tespiti doğrulaması."""
        cap = cv2.VideoCapture('test_verileri/golge.webm')
        self.assertTrue(cap.isOpened(), "golge.webm açılamadı")
        for _ in range(5):
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, (640, 480))
            sonuc = self.dedektor.tespit_et(frame)
            self.assertIsNotNone(sonuc['sol_cizgi'])
            self.assertIsNotNone(sonuc['sag_cizgi'])
        cap.release()

    def test_pid_hesapla_kolaylik_metodu(self):
        """PIDSurucu.hesapla metodu tek adımda (aci, hiz, durum) döndürmeli."""
        pid = PIDSurucu()
        aci, hiz, durum = pid.hesapla(0.0)
        self.assertEqual(aci, 0.0)
        self.assertGreater(hiz, 0)
        self.assertIn("DENGELI", durum)


if __name__ == '__main__':
    unittest.main()
