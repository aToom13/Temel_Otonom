import argparse
import cv2
import numpy as np


def bos_fonksiyon(x):
    pass


def main():
    parser = argparse.ArgumentParser(description="Canli HSV serit renk filtresi kalibrasyon araci")
    parser.add_argument("--kaynak", default="0", help="Kamera ID (0, 1) veya video dosya yolu")
    args = parser.parse_args()

    kaynak = int(args.kaynak) if args.kaynak.isdigit() else args.kaynak
    cap = cv2.VideoCapture(kaynak)

    if not cap.isOpened():
        print(f"[HATA] Kaynak acilamadi: {kaynak}")
        return

    cv2.namedWindow("HSV Kalibrasyon")
    cv2.createTrackbar("Alt-H", "HSV Kalibrasyon", 0, 255, bos_fonksiyon)
    cv2.createTrackbar("Alt-S", "HSV Kalibrasyon", 0, 255, bos_fonksiyon)
    cv2.createTrackbar("Alt-V", "HSV Kalibrasyon", 180, 255, bos_fonksiyon)
    cv2.createTrackbar("Ust-H", "HSV Kalibrasyon", 255, 255, bos_fonksiyon)
    cv2.createTrackbar("Ust-S", "HSV Kalibrasyon", 80, 255, bos_fonksiyon)
    cv2.createTrackbar("Ust-V", "HSV Kalibrasyon", 255, 255, bos_fonksiyon)

    print("Cikmak icin 'q' tusuna basin.")

    while True:
        ret, kare = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, kare = cap.read()
            if not ret:
                break

        kare = cv2.resize(kare, (640, 480))
        hsv = cv2.cvtColor(kare, cv2.COLOR_BGR2HSV)

        alt_h = cv2.getTrackbarPos("Alt-H", "HSV Kalibrasyon")
        alt_s = cv2.getTrackbarPos("Alt-S", "HSV Kalibrasyon")
        alt_v = cv2.getTrackbarPos("Alt-V", "HSV Kalibrasyon")
        ust_h = cv2.getTrackbarPos("Ust-H", "HSV Kalibrasyon")
        ust_s = cv2.getTrackbarPos("Ust-S", "HSV Kalibrasyon")
        ust_v = cv2.getTrackbarPos("Ust-V", "HSV Kalibrasyon")

        alt_sinir = np.array([alt_h, alt_s, alt_v])
        ust_sinir = np.array([ust_h, ust_s, ust_v])

        maske = cv2.inRange(hsv, alt_sinir, ust_sinir)
        filtrelenmis = cv2.bitwise_and(kare, kare, mask=maske)

        cv2.imshow("HSV Kalibrasyon - Orijinal", kare)
        cv2.imshow("HSV Kalibrasyon - Maske", maske)
        cv2.imshow("HSV Kalibrasyon - Filtreli", filtrelenmis)

        tus = cv2.waitKey(25) & 0xFF
        if tus == ord('q') or tus == 27:
            print("\n--- config.yaml icin HSV sinirlari ---")
            print("hsv_filtre:")
            print(f"  alt: [{alt_h}, {alt_s}, {alt_v}]")
            print(f"  ust: [{ust_h}, {ust_s}, {ust_v}]\n")
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
