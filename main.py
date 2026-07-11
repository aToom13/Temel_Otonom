import argparse
import time
import cv2
import yaml

from core import KlasikSeritDedektoru, SeritDedektoru, YOLODedektoru, YOLOSeritDedektoru, PIDSurucu
from araclar import Gorsellestirici
from donanim import ArduinoHaberlesme


def yapilandirma_yukle(yol="config.yaml"):
    with open(yol, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Temel Otonom Sürüş ve Şerit Takip Sistemi")
    parser.add_argument("--video", type=str, default="test_verileri/otoban.mp4", help="Giriş video dosyası")
    parser.add_argument("--kamera", type=int, default=None, help="Canlı kamera cihaz ID (örn: 0)")
    parser.add_argument("--mod", type=str, choices=["klasik", "kus_bakisi", "yolo", "hibrit", "yolo_serit"], default="hibrit",
                        help="Çalışma modu: klasik, kus_bakisi, yolo, hibrit veya yolo_serit")
    parser.add_argument("--debug", "--panel", dest="debug", action="store_true",
                        help="Kuş bakışı ve maske içeren çoklu panel modunu açar")
    parser.add_argument("--seri", action="store_true", help="Gerçek Arduino seri port bağlantısını açar")
    parser.add_argument("--simulasyon", action="store_true", default=None, help="Donanımsız simüle mod")
    parser.add_argument("--protokol", type=str, choices=["ascii_ha", "kat4_rv"], default=None,
                        help="Arduino protokolü: ascii_ha (H110A-10) veya kat4_rv (r015v110)")
    parser.add_argument("--cikti", type=str, default=None, help="İşlenmiş videoyu kaydetmek için dosya yolu")
    parser.add_argument("--maks-kare", type=int, default=None, help="İşlenecek maksimum kare sayısı (testler için)")
    parser.add_argument("--headless", action="store_true", help="GUI penceresi açmadan çalışır")
    args = parser.parse_args()

    cfg = yapilandirma_yukle()
    w = cfg["kamera"]["genislik"]
    h = cfg["kamera"]["yukseklik"]

    # Simülasyon bayrağını belirle
    simulasyon_aktif = True if args.simulasyon is True else (not args.seri)
    protokol = args.protokol or cfg.get("seri_port", {}).get("protokol", "ascii_ha")

    # Modülleri başlat
    klasik_serit = KlasikSeritDedektoru(genislik=w, yukseklik=h)
    kus_bakisi_serit = SeritDedektoru(genislik=w, yukseklik=h, perspektif=cfg.get("perspektif"))
    yolo_dedektor = YOLODedektoru(
        model_yolu=cfg["yolo"]["model_yolu"],
        guven_esigi=cfg["yolo"]["guven_esigi"],
        odak_uzakligi=cfg["yolo"].get("odak_uzakligi", 950.0)
    )
    yolo_serit = YOLOSeritDedektoru(
        model_yolu=cfg.get("yolo_serit", {}).get("model_yolu", "modeller/yolov8_serit_seg.pt"),
        guven_esigi=cfg.get("yolo_serit", {}).get("guven_esigi", 0.30),
        genislik=w,
        yukseklik=h
    )
    pid = PIDSurucu(
        kp=cfg["pid"]["kp"],
        ki=cfg["pid"]["ki"],
        kd=cfg["pid"]["kd"],
        max_aci=cfg["pid"]["max_direksiyon_acisi"],
        taban_hiz=cfg["hiz"]["taban_hiz"],
        min_hiz=cfg["hiz"]["min_hiz"]
    )
    gorsel = Gorsellestirici(genislik=w, yukseklik=h)
    arduino = ArduinoHaberlesme(
        port=cfg["seri_port"]["port"],
        baudrate=cfg["seri_port"]["baudrate"],
        simulasyon=simulasyon_aktif,
        protokol=protokol
    )

    # Video kaynağı
    kaynak = args.kamera if args.kamera is not None else args.video
    cap = cv2.VideoCapture(kaynak)

    if not cap.isOpened():
        print(f"[HATA] Video kaynağı açılamadı: {kaynak}")
        return

    writer = None
    if args.cikti:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_w = int(w * (16 / 9) + w * (4 / 3) * 0.5) if args.debug else w
        writer = cv2.VideoWriter(args.cikti, fourcc, 30.0, (out_w, h))

    print("=" * 65)
    print("  TEMEL OTONOM SÜRÜŞ SİSTEMİ BAŞLATILDI")
    print(f"  Mod: {args.mod.upper()} | Kaynak: {kaynak} | Çoklu Panel: {args.debug}")
    print(f"  Donanım: {'CANLI ARDUINO' if not simulasyon_aktif else 'SİMÜLASYON'} (Protokol: {protokol})")
    print("=" * 65)

    onceki_zaman = time.time()
    kare_sayaci = 0

    try:
        while True:
            ret, kare = cap.read()
            if not ret:
                # Video bittiğinde başa sar (canlı kamera değilse)
                if args.kamera is None and args.maks_kare is None:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, kare = cap.read()
                    if not ret:
                        break
                else:
                    break

            kare = cv2.resize(kare, (w, h))
            simdiki_zaman = time.time()
            fps = 1.0 / (simdiki_zaman - onceki_zaman) if (simdiki_zaman - onceki_zaman) > 0 else 30.0
            onceki_zaman = simdiki_zaman

            # 1. Şerit Tespiti
            serit_bilgisi = None
            kus_bakisi = None
            maske = None
            egrilik = None

            if args.mod in ["klasik", "hibrit"]:
                serit_bilgisi = klasik_serit.tespit_et(kare)
                kus_bakisi, maske, kb_bilgi = kus_bakisi_serit.isle(kare)
                sapma = serit_bilgisi["sapma_piksel"]
                egrilik = kb_bilgi.get("egrilik_m")
            elif args.mod == "kus_bakisi":
                kus_bakisi, maske, kb_bilgi = kus_bakisi_serit.isle(kare)
                serit_bilgisi = kb_bilgi
                sapma = kb_bilgi["sapma_piksel"]
                egrilik = kb_bilgi.get("egrilik_m")
            elif args.mod == "yolo_serit":
                serit_bilgisi = yolo_serit.tespit_et(kare)
                sapma = serit_bilgisi["sapma_piksel"]
                maske = serit_bilgisi["maske"]
                kus_bakisi = cv2.cvtColor(maske, cv2.COLOR_GRAY2BGR)
            else:
                sapma = 0.0

            # 2. Nesne, Tabela ve Araç Tespiti
            tespitler = []
            if args.mod in ["yolo", "hibrit"]:
                tespitler = yolo_dedektor.tespit_et(kare)

            # 3. PID Karar Mekanizması
            direksiyon_acisi = pid.direksiyon_hesapla(sapma)
            hiz, durum = pid.hiz_belirle(direksiyon_acisi, tespitler)

            # 4. Donanım / Arduino Komut İletimi
            arduino.komut_gonder(hiz, direksiyon_acisi)

            # 5. Görselleştirme: Şerit Koridoru + Araç Gidiş Yolu (Trajektori Bandı)
            if serit_bilgisi:
                islenmis_kare = gorsel.serit_koridoru_ciz(kare, serit_bilgisi)
            else:
                islenmis_kare = kare.copy()

            # Algılanan nesneleri ve mesafeleri çiz
            for t in tespitler:
                x1, y1, x2, y2 = t["kutu"]
                etiket = t["etiket"]
                mesafe = t.get("mesafe", 0)
                cv2.rectangle(islenmis_kare, (x1, y1), (x2, y2), (0, 235, 255), 2)
                cv2.putText(islenmis_kare, f"{etiket} {mesafe:.1f}m", (x1, y1 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 235, 255), 1, cv2.LINE_AA)

            en_yakin = f"{tespitler[0]['etiket']}: {tespitler[0]['mesafe']:.1f}m" if tespitler else None
            islenmis_kare = gorsel.telemetri_paneli_ciz(
                islenmis_kare, direksiyon_acisi, hiz, durum, sapma, fps, en_yakin, egrilik
            )

            if args.debug and kus_bakisi is not None and maske is not None:
                gosterim = gorsel.coklu_panel_olustur(islenmis_kare, kus_bakisi, maske)
            else:
                gosterim = islenmis_kare

            if writer is not None:
                writer.write(cv2.resize(gosterim, (writer.get(cv2.CAP_PROP_FRAME_WIDTH), h)) if args.debug else gosterim)

            if not args.headless:
                cv2.imshow("Temel Otonom", gosterim)
                tus = cv2.waitKey(1) & 0xFF
                if tus == ord("q") or tus == 27:
                    break

            kare_sayaci += 1
            if args.maks_kare is not None and kare_sayaci >= args.maks_kare:
                break

    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if not args.headless:
            cv2.destroyAllWindows()
        arduino.kapat()
        print(f"[BILGI] Sistem kapatıldı. Toplam işlenen kare: {kare_sayaci}")


if __name__ == "__main__":
    main()
