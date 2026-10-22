# Temel Otonom Sürüş ve Görüntü İşleme Sistemi

Python, OpenCV ve YOLOv8 ile kamera görüntüsünden şerit ve nesne tespiti, yaklaşık mesafe kestirimi ve PID tabanlı direksiyon komutları üreten deneysel bir sürüş projesi. Video dosyalarıyla donanımsız çalıştırılabilir; Arduino bağlantısıyla motor ve servo komutları gönderebilir.

Görüntü işleme, kontrol algoritmaları ve seri haberleşmeyi aynı akışta denemek için geliştirilmiştir.

---

## Ekran Görüntüleri ve Canlı Arayüz

| Teknofest Robotaksi Pisti (HUD & Trajektori) | Otoban Sürüşü & Araç Takibi (HUD) |
| :---: | :---: |
| ![HUD Görünümü](gorseller/hud_gorunumu.jpg) | ![Otoban HUD](gorseller/otoban_hud.jpg) |

| Geliştirici Çoklu Görünüm Paneli | Kuş Bakışı (IPM & Kayan Pencere) |
| :---: | :---: |
| ![Çoklu Panel](gorseller/coklu_panel.jpg) | ![Kuş Bakışı](gorseller/kus_bakisi.jpg) |

---

## Kurulum ve Çalıştırma

Python 3 ve GUI kullanımı için masaüstü oturumu gerekir. Komutları proje kökünden çalıştırın.

### 1. Bağımlılıkların Yüklenmesi

```bash
git clone https://github.com/aToom13/Temel_Otonom.git
cd Temel_Otonom
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Sistemi Çalıştırma

Varsayılan olarak seri bağlantı kapalıdır; `--seri` gerçek donanıma komut gönderimini açar. Önce video ile deneyin. Model yolları, kamera, PID ve seri port ayarları [config.yaml](config.yaml) dosyasındadır.

```bash
# Hibrit Mod (Klasik Şerit + YOLO Araç/Mesafe Takibi + HUD Trajektori)
python main.py --mod hibrit --video test_verileri/otoban.mp4

# Geliştirici / Debug Modu (Kuş Bakışı ve İkili Maske Paneli Dahil)
python main.py --mod hibrit --video test_verileri/otoban.mp4 --debug

# Saf Klasik Görüntü İşleme Modu (Canny + Hough)
python main.py --mod klasik --video test_verileri/otoban.mp4

# Kuş Bakışı Kayan Pencere Modu (IPM + 2. Derece Polinom)
python main.py --mod kus_bakisi --video test_verileri/golge.webm

# Canlı USB Kamera ile Çalıştırma
python main.py --kamera 0 --mod hibrit

# YOLOv8 Semantik Şerit Takibi Modu (Ağaç & Çadır Gölgelerine Dayanıklı)
python main.py --mod yolo_serit --video test_verileri/golge.webm

# Bağımsız YOLO Şerit Test Scripti
python yolo_serit_test.py --video test_verileri/golge.webm

# Arduino Donanım Bağlantısı ile Çalıştırma (Kat4 Protokolü)
python main.py --video test_verileri/otoban.mp4 --seri --protokol kat4_rv
```

### 3. Kalibrasyon Araçları

* **Perspektif ROI Kalibrasyonu:**
  ```bash
  python araclar/roi_secici.py --kaynak test_verileri/otoban.mp4
  ```
  *(Sırasıyla: Sol-Üst, Sol-Alt, Sağ-Üst, Sağ-Alt noktalarına tıklayın; `r` ile sıfırlayın).*

* **HSV Renk Eşik Kalibrasyonu:**
  ```bash
  python araclar/hsv_ayar.py --kaynak test_verileri/otoban.mp4
  ```

GUI açmadan sınırlı sayıda kare işlemek için:

```bash
python main.py --mod hibrit --video test_verileri/otoban.mp4 --headless --maks-kare 120 --simulasyon
```

Gerçek donanım için önce `config.yaml` içindeki port ve baudrate değerlerini, Arduino yazılımını ve seçili protokolü eşleştirin.

---

## Adım Adım Mimari ve Evrim Süreci

Proje, geleneksel görüntü işlemeden başlayıp derin öğrenme ve donanım entegrasyonuna uzanan 4 kademeli bir evrim mimarisine sahiptir:

### 1. Aşama: Geleneksel Görüntü İşleme (Canny + Hough + Polinom)
* **Ön İşleme & Gölge Dayanımı:** HLS ve HSV renk uzayına dönüştürme, **CLAHE** (Contrast Limited Adaptive Histogram Equalization) ile sert gölgeli ve aşırı aydınlık zeminlerin normalizasyonu.
* **Bariyer Filtresi:** Teknofest ve pist testlerinde kenarlardaki kırmızı/turuncu plastik bariyerlerin şerit olarak algılanmasını önleyen doygunluk ve ton filtrelemesi.
* **İlgi Alanı (ROI):** Yol dışı unsurlar (gökyüzü, kaput, ağaçlar) dinamik yamuk poligon ile maskelenir.
* **Hough Çizgi Dönüşümü:** `HoughLinesP` ile kenar pikselleri çizgi vektörlerine dönüştürülür, eğimlerine göre ($\frac{\Delta y}{\Delta x}$) sol ve sağ şerit doğruları olarak gruplanır.
* **Şerit Koridoru & Araç Gidiş Yolu (Trajectory Ribbon):** Şerit sınırları arasına zümrüt yeşili saydam sürüş koridoru yerleştirilir; şeridin tam ortasına ise ileri yönlü navigasyon okları (`^`), neon trajektori bandı ve araç hedefleme odak noktası içeren parlak yörünge çizgisi çizilir.

### 2. Aşama: Kuş Bakışı Dönüşümü (IPM) & Kayan Pencere (Sliding Window)
* **Ters Perspektif Eşleme (IPM):** 4 noktalı homografi matrisi ile kamera açısı yukarıdan dik kuş bakışı açısına dönüştürülür (`cv2.warpPerspective`).
* **Histogram Tabanlı Kayan Pencereler:** Histogram piksel tepelerine göre dikeyde 10 pencere boyunca şerit pikselleri toplanır ve 2. derece parabolik eğri uydurulur:
  $$x = A \cdot y^2 + B \cdot y + C$$
* **Viraj Yarıçapı (Eğrilik):** Yol eğriliği pikselden metreye dönüştürülerek hesaplanır:
  $$R_{\text{curve}} = \frac{\left(1 + (2Ay + B)^2\right)^{3/2}}{|2A|}$$

### 3. Aşama: Derin Öğrenme / YOLOv8 ve Mesafe Kestirimi
* **Güneş - Gölge Problemi & YOLOv8 Şerit Segmentasyonu:** Geleneksel HSV renk filtreleri, yarışma parkurlarındaki ağaç ve çadır gölgelerinde ($V$ parlaklık kanalının ani düşmesi) şerit çizgilerini kaybeder veya güneş altındaki açık renkli asfaltı/bariyerleri şerit zannederek hataya düşer. Bu sorunu çözmek için eğitilen **YOLOv8 Şerit Segmentasyon Modeli (`modeller/yolov8_serit_seg.pt`)** ve [`core/yolo_serit.py`](core/yolo_serit.py), renk eşiklerine alternatif bir şerit segmentasyonu yaklaşımı sunar. Gölge içeren örnek kayıt `test_verileri/golge.webm` dosyasındadır; farklı ışık ve yol koşulları için ayrıca değerlendirme gerekir.
* **Nesne & Araç Tespiti:** YOLOv8 mimarisi ile öndeki araçlar (`car`, `truck`, `bus`) ve trafik işaretleri eş zamanlı taranır.
* **Pinhole Kamera Mesafe Kestirimi:** Bounding box piksel genişliği ($w_{px}$), kameranın odak uzaklığı ($f$) ve aracın gerçek dünyadaki ortalama genişliği ($W \approx 1.85\text{ m}$) kullanılarak araca olan yaklaşık mesafe kestirilir:
  $$\text{Mesafe (m)} = \frac{W_{\text{gerçek}} \times f}{w_{\text{kutu}}}$$

### 4. Aşama: PID Kontrol & Donanım Haberleşmesi
* **Merkez Sapma:** Aracın kamera merkezi ile şerit merkezi arasındaki piksel farkı ($e$) hesaplanır.
* **Anti-Windup PID Direksiyon:** Doyum korumalı PID algoritması ile servo dönüş açısı ($-35^\circ$ ile $+35^\circ$ arası) üretilir:
  $$u(t) = K_p \cdot e(t) + K_i \int e(t)dt + K_d \cdot \frac{de(t)}{dt}$$
* **Akıllı Hız Kontrolü:** Öndeki araç takip mesafesi ($<8\text{ m}$) veya viraj keskinliğine göre motor PWM değeri dinamik olarak azaltılır; yakın engel veya durma koşullarında sıfır hız komutu üretilir. Bu davranış fiziksel duruş garantisi değildir.
* **Çift Arduino Seri Protokolü:**
  1. *Standart ASCII Protokolü:* `H<hiz>A<aci>\n` (Örnek: `H110A-12.5\n`)
  2. *Kat4 Teknofest Robotaksi Protokolü:* `[f/r/l]<aci>v<hiz>\n` (Örnek: `r015v110\n`, `l020v095\n`, `f000v120\n`)

---

## Proje Dizin Yapısı

```text
Temel_Otonom/
├── core/
│   ├── __init__.py
│   ├── klasik_serit.py       # Faz 1: Canny + Hough + Polinom şerit tespiti
│   ├── serit_tespit.py       # Faz 2: IPM kuş bakışı & sliding window tespit modülü
│   ├── yolo_tespit.py        # Faz 3: YOLOv8 nesne, araç ve tabela tespiti
│   ├── yolo_serit.py         # Faz 3: YOLOv8 şerit segmentasyonu (Gölgeye dayanıklı şerit takibi)
│   ├── mesafe_hesaplayici.py # Bounding box boyutundan mesafe kestirim motoru
│   └── pid_surucu.py         # Faz 4: Anti-windup PID direksiyon ve akıllı hız regülatörü
├── donanim/
│   ├── __init__.py
│   ├── arduino_haberlesme.py # USB UART seri port iletişimi (Çift protokol & simülasyon destekli)
│   └── otonom_surucu.ino     # DC motor ve yön servosu için Arduino firmware
├── araclar/
│   ├── __init__.py
│   ├── cizici.py             # HUD telemetri paneli, şerit koridoru ve gidiş yolu çizici
│   ├── roi_secici.py         # Perspektif köşe noktalarını kalibre etme aracı
│   └── hsv_ayar.py           # Canlı trackbar'lar ile HSV filtre kalibrasyon aracı
├── modeller/
│   ├── best.pt               # YOLOv8 nesne, araç ve tabela tespit ağırlıkları
│   └── yolov8_serit_seg.pt   # YOLOv8 semantik şerit segmentasyon model ağırlıkları
├── yolo_serit_test.py        # Gölge parkurları için YOLOv8 şerit takip test scripti
├── test_verileri/
│   ├── otoban.mp4            # Otoyol şerit ve araç sürüş test kaydı
│   ├── golge.webm            # Gölge varyasyonlu pist sürüş kaydı
│   └── yeni.webm             # Viraj ve tabela pist sürüş kaydı
├── gorseller/
│   ├── hud_gorunumu.jpg      # Canlı HUD ve araç gidiş yolu ekran görüntüsü
│   ├── coklu_panel.jpg       # Geliştirici çoklu kamera / debug görünümü
│   ├── kus_bakisi.jpg        # Kuş bakışı perspektif çizimi
│   └── hsv_maske.jpg         # İkili şerit kenar maskesi
├── tests/
│   └── test_pipeline.py      # Kapsamlı birim ve entegrasyon test paketi
├── config.yaml               # Sistem parametreleri (PID, Kamera, YOLO, ROI)
├── main.py                   # Modüler ana çalıştırma giriş noktası
├── requirements.txt          # Python bağımlılıkları
└── README.md                 # Kapsamlı teknik dokümantasyon
```

---

## Donanım Bağlantı Şeması ve Seri Protokol

### Arduino Pin Atamaları
| Donanım | Arduino Pini | Sinyal Tipi | Açıklama |
| :--- | :--- | :--- | :--- |
| **Yön Servosu** | Pin 9 | PWM (Servo) | 55° (Sol) - 90° (Merkez) - 125° (Sağ) |
| **DC Motor Hızı** | Pin 5 | PWM | 0 - 255 PWM Sürüş Hızı |
| **DC Motor Yönü** | Pin 6 | Dijital OUT | HIGH = İleri, LOW = Geri / Fren |

### Seri Protokol Formatları
1. **ASCII Protokolü (`ascii_ha`):**
   ```text
   H<hiz>A<aci>\n
   Örnek: H110A-14.5\n  -> Hız 110 PWM, Direksiyon -14.5 derece (Sola)
   ```
2. **Kat4 Robotaksi Protokolü (`kat4_rv`):**
   ```text
   [f/r/l]<aci_3_basamak>v<hiz_3_basamak>\n
   Örnek: r015v110\n    -> Sağa 15 derece, Hız 110 PWM
   Örnek: l020v095\n    -> Sola 20 derece, Hız 95 PWM
   Örnek: f000v120\n    -> Düz gidiş, Hız 120 PWM
   ```

---

## Birim ve Entegrasyon Testleri

Mevcut test paketi [tests/test_pipeline.py](tests/test_pipeline.py) dosyasındadır:

**Bilinen uyumsuzluk:** Test dosyası `ArduinoBaglanti` adını içe aktarıyor; güncel modüldeki sınıf `ArduinoHaberlesme`. Test paketinin Arduino API’sine uyarlanması gerekir. Test giriş noktası:

```bash
python -m unittest tests/test_pipeline.py -v
```

---

## Sınırlar

- Mesafe ve eğrilik değerleri kamera kalibrasyonu ile varsayılan fiziksel boyutlara bağlıdır.
- Şerit ve nesne modellerinin başarımı veri setine, ışığa ve kamera açısına göre değişir.
- Gerçek araç denemeleri kontrollü ortam ve bağımsız acil durdurma düzeni gerektirir.

## Yazar ve lisans durumu

- **Geliştirici:** aToom13 (<200818akif@gmail.com>)
- Depoda henüz ayrı bir `LICENSE` dosyası bulunmuyor.

