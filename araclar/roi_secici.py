import argparse
import cv2
import numpy as np


noktalar = []


def fare_olayi(event, x, y, flags, param):
    global noktalar
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(noktalar) < 4:
            noktalar.append((x, y))
            print(f"Secilen Nokta {len(noktalar)}: ({x}, {y})")


def main():
    parser = argparse.ArgumentParser(description="Kus bakisi perspektif donusum noktalarini secme araci")
    parser.add_argument("--kaynak", default="0", help="Kamera ID (0, 1) veya video dosya yolu")
    args = parser.parse_args()

    kaynak = int(args.kaynak) if args.kaynak.isdigit() else args.kaynak
    cap = cv2.VideoCapture(kaynak)

    if not cap.isOpened():
        print(f"[HATA] Kaynak acilamadi: {kaynak}")
        return

    ret, kare = cap.read()
    if not ret:
        print("[HATA] Kare alinamadi.")
        return

    kare = cv2.resize(kare, (640, 480))
    pencere_adi = "ROI Secici - Sirayla: Sol-Ust, Sol-Alt, Sag-Ust, Sag-Alt tiklayin"
    cv2.namedWindow(pencere_adi)
    cv2.setMouseCallback(pencere_adi, fare_olayi)

    print("\nLutfen sirasiyla su noktalara tiklayin:")
    print("1. Sol Ust (Uzak)")
    print("2. Sol Alt (Yakin)")
    print("3. Sag Ust (Uzak)")
    print("4. Sag Alt (Yakin)")
    print("Cikmak icin 'q' veya ESC tusuna basin.\n")

    while True:
        ekran = kare.copy()

        for idx, pt in enumerate(noktalar):
            cv2.circle(ekran, pt, 5, (0, 255, 0), -1)
            cv2.putText(ekran, f"P{idx+1}", (pt[0] + 10, pt[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        if len(noktalar) == 4:
            # Poligon ciz
            cizgi_sirasi = np.array([noktalar[0], noktalar[1], noktalar[3], noktalar[2]], np.int32)
            cv2.polylines(ekran, [cizgi_sirasi], True, (255, 0, 0), 2)

        cv2.imshow(pencere_adi, ekran)
        tus = cv2.waitKey(20) & 0xFF
        if tus == ord('q') or tus == 27:
            break
        elif tus == ord('r'):
            noktalar.clear()
            print("Noktalar sifirlandi.")

    cap.release()
    cv2.destroyAllWindows()

    if len(noktalar) == 4:
        print("\n--- config.yaml icin koordinatlar ---")
        print("perspektif:")
        print(f"  sol_ust: [{noktalar[0][0]}, {noktalar[0][1]}]")
        print(f"  sol_alt: [{noktalar[1][0]}, {noktalar[1][1]}]")
        print(f"  sag_ust: [{noktalar[2][0]}, {noktalar[2][1]}]")
        print(f"  sag_alt: [{noktalar[3][0]}, {noktalar[3][1]}]\n")


if __name__ == "__main__":
    main()
