#!/usr/bin/env python3
"""
YOLOv8 ile Şerit Takibi ve Gölge Dayanım Testi.

Bu script; Teknofest ve gerçek dünya sürüşlerinde ağaçlar, binalar ve çadırların
oluşturduğu sert gölgelerde klasik HSV renk filtrelemesinin yetersiz kalması
nedeniyle eğitilen YOLOv8 Lane Segmentation modelinin performansını doğrulamak
için geliştirilmiştir.

Kullanım:
    python yolo_serit_test.py --video test_verileri/golge.webm --headless --maks-kare 30
    python yolo_serit_test.py --video test_verileri/otoban.mp4 --maks-kare 50
"""

import argparse
import time
import cv2
import numpy as np

from core.yolo_serit import YOLOSeritDedektoru
from core.pid_surucu import PIDSurucu
from araclar.cizici import Gorsellestirici


def main():
    parser = argparse.ArgumentParser(description="YOLOv8 Tabanlı Şerit Takip ve Gölge Testi")
    parser.add_argument("--video", type=str, default="test_verileri/golge.webm",
                        help="Test edilecek video dosyası (Varsayılan: test_verileri/golge.webm)")
    parser.add_argument("--model", type=str, default="modeller/yolov8_serit_seg.pt",
                        help="YOLOv8 şerit segmentasyon model dosyası")
    parser.add_argument("--maks-kare", type=int, default=None,
                        help="İşlenecek maksimum kare sayısı")
    parser.add_argument("--cikti", type=str, default=None,
                        help="İşlenmiş çıktı videosunun kaydedileceği dosya")
    parser.add_argument("--kaydet-ornek", type=str, default=None,
                        help="İlk başarılı kareyi görsel olarak kaydetme yolu")
    parser.add_argument("--headless", action="store_true",
                        help="Pencere açmadan arka planda çalıştır")
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"[HATA] Video dosyası açılamadı: {args.video}")
        return

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_in = cap.get(cv2.CAP_PROP_FPS) or 30.0

    print("=" * 65)
    print("  YOLOv8 ŞERİT TAKİP VE GÖLGE TEST MOTORU BAŞLATILDI")
    print(f"  Kaynak: {args.video} ({w}x{h} @ {fps_in:.1f} FPS)")
    print(f"  Model: {args.model}")
    print("  Amaç: Sert gölgeli zeminlerde HSV aşımını ve şerit sürekliliğini test etmek")
    print("=" * 65)

    dedektor = YOLOSeritDedektoru(model_yolu=args.model, genislik=w, yukseklik=h)
    pid = PIDSurucu(kp=0.35, ki=0.02, kd=0.15, max_aci=28.0)
    cizici = Gorsellestirici(genislik=w, yukseklik=h)

    writer = None
    if args.cikti:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.cikti, fourcc, fps_in, (w, h))

    kare_sayaci = 0
    t_baslangic = time.time()
    son_zaman = time.time()

    while True:
        ret, kare = cap.read()
        if not ret:
            break

        kare_sayaci += 1
        t_simdi = time.time()
        dt = max(t_simdi - son_zaman, 0.001)
        son_zaman = t_simdi

        # 1. YOLO Tabanlı Şerit Tespiti
        sonuc = dedektor.tespit_et(kare)
        sapma = sonuc["sapma_piksel"]

        # 2. PID Kontrolü
        aci, hiz, durum = pid.hesapla(sapma, dt=dt)

        # 3. Görselleştirme (Koridor, Trajektori ve HUD)
        islenmis = cizici.serit_koridoru_ciz(kare, sonuc)

        fps_anlik = 1.0 / dt
        islenmis = cizici.telemetri_paneli_ciz(
            islenmis,
            aci=aci,
            hiz=hiz,
            durum=f"YOLO-SEG: {durum}",
            sapma=sapma,
            fps=fps_anlik,
            nesne_bilgisi="YOLOv8 Semantik Serit"
        )

        if writer:
            writer.write(islenmis)

        if args.kaydet_ornek and kare_sayaci == 1:
            cv2.imwrite(args.kaydet_ornek, islenmis)
            print(f"[BILGI] Örnek görsel kaydedildi: {args.kaydet_ornek}")

        if not args.headless:
            cv2.imshow("YOLOv8 Serit Takip Testi", islenmis)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        if args.maks_kare and kare_sayaci >= args.maks_kare:
            break

    sure = time.time() - t_baslangic
    cap.release()
    if writer:
        writer.release()
    if not args.headless:
        cv2.destroyAllWindows()

    ortalama_fps = kare_sayaci / max(sure, 0.001)
    print(f"\n[BASARILI] Test tamamlandı. {kare_sayaci} kare işlendi. Ortalama Hız: {ortalama_fps:.1f} FPS.")


if __name__ == "__main__":
    main()
