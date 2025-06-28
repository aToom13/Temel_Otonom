# Dursun Otonom Navigasyon Sistemi

<div align="center">

![Dursun Logo](https://via.placeholder.com/200x100/4CAF50/FFFFFF?text=DURSUN)

**Gelişmiş Bilgisayar Görüşü ve Otonom Navigasyon Platformu**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![React](https://img.shields.io/badge/React-18+-61DAFB.svg)](https://reactjs.org)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-green.svg)](https://opencv.org)
[![YOLO](https://img.shields.io/badge/YOLO-v8-orange.svg)](https://ultralytics.com)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

</div>

## 📋 İçindekiler

- [Proje Hakkında](#-proje-hakkında)
- [Özellikler](#-özellikler)
- [Sistem Gereksinimleri](#-sistem-gereksinimleri)
- [Kurulum](#-kurulum)
- [Kullanım](#-kullanım)
- [Proje Yapısı](#-proje-yapısı)
- [API Dokümantasyonu](#-api-dokümantasyonu)
- [Konfigürasyon](#-konfigürasyon)
- [Test](#-test)
- [Sorun Giderme](#-sorun-giderme)
- [Katkıda Bulunma](#-katkıda-bulunma)
- [Lisans](#-lisans)

## 🚀 Proje Hakkında

Dursun, gelişmiş bilgisayar görüşü teknolojileri kullanarak otonom navigasyon sağlayan kapsamlı bir platformdur. ZED 2i stereo kamera, YOLO nesne algılama, şerit takibi ve Arduino tabanlı donanım kontrolü ile gerçek zamanlı otonom sürüş yetenekleri sunar.

### 🎯 Temel Hedefler

- **Güvenli Navigasyon**: Gelişmiş engel algılama ve kaçınma
- **Gerçek Zamanlı İşleme**: Düşük gecikme ile yüksek performans
- **Modüler Mimari**: Kolay genişletilebilir ve özelleştirilebilir
- **Web Tabanlı Kontrol**: Modern ve kullanıcı dostu arayüz

## ✨ Özellikler

### 🎥 Görüntü İşleme
- **ZED 2i Stereo Kamera Desteği**: Derinlik haritası ve 3D algılama
- **Webcam Fallback**: ZED kamera yokken otomatik geçiş
- **YOLO v8 Nesne Algılama**: Trafik işaretleri ve araçlar
- **Şerit Algılama**: Gerçek zamanlı yol takibi
- **Derinlik Analizi**: 3D engel algılama ve mesafe ölçümü

### 🧠 Yapay Zeka ve Algoritmalar
- **PID Kontrolcü**: Hassas direksiyon ve hız kontrolü
- **Polinom Şerit Uydurma**: Eğrisel yol takibi
- **Blob Algılama**: Özel nesne tanıma
- **Histogram Analizi**: Görüntü kalitesi değerlendirmesi
- **ROI (İlgi Alanı) Optimizasyonu**: Performans artırımı

### 🔧 Donanım Entegrasyonu
- **Arduino İletişimi**: Serial port üzerinden komut gönderimi
- **Motor Kontrolü**: PWM tabanlı hız kontrolü
- **Servo Kontrolü**: Hassas direksiyon kontrolü
- **Sensor Entegrasyonu**: Genişletilebilir sensor desteği

### 🌐 Web Arayüzü
- **React SPA**: Modern ve responsive tasarım
- **Gerçek Zamanlı Video**: WebSocket ile canlı görüntü
- **Telemetri Paneli**: Detaylı sistem durumu
- **Kontrol Paneli**: Manuel müdahale imkanı
- **Log İzleme**: Sistem olaylarını takip

### 📊 İzleme ve Analiz
- **Sistem Sağlığı**: Bileşen durumu izleme
- **Performans Metrikleri**: FPS, işleme süresi, bellek kullanımı
- **Hata Yönetimi**: Kapsamlı loglama ve hata raporlama
- **Thread Supervisor**: Otomatik yeniden başlatma

## 💻 Sistem Gereksinimleri

### Minimum Gereksinimler
- **İşletim Sistemi**: Windows 10/11, Ubuntu 18.04+, macOS 10.15+
- **Python**: 3.8 veya üzeri
- **RAM**: 4GB (8GB önerilen)
- **Depolama**: 2GB boş alan
- **USB Port**: Arduino bağlantısı için

### Önerilen Gereksinimler
- **İşlemci**: Intel i5 8. nesil veya AMD Ryzen 5 3600
- **RAM**: 8GB DDR4
- **GPU**: NVIDIA GTX 1060 veya üzeri (CUDA desteği)
- **ZED 2i Kamera**: Stereo görüş için
- **SSD**: Hızlı veri erişimi için

### Desteklenen Donanım
- **Kameralar**: ZED 2i, ZED 2, ZED Mini, USB Webcam
- **Mikrocontroller**: Arduino Uno, Nano, Mega
- **Sensörler**: Ultrasonik, LiDAR, IMU (gelecek sürümlerde)

## 🛠 Kurulum

### 1. Depoyu Klonlayın
```bash
git clone https://github.com/kullanici/dursun-project.git
cd dursun-project
```

### 2. Python Ortamını Hazırlayın
```bash
# Sanal ortam oluşturun
python -m venv venv

# Sanal ortamı aktifleştirin
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

# Bağımlılıkları yükleyin
pip install -r requirements.txt
```

### 3. ZED SDK Kurulumu (Opsiyonel)
```bash
# ZED SDK'yı indirin ve kurun
# https://www.stereolabs.com/developers/release/
# Python API'yi yükleyin
pip install pyzed
```

### 4. Node.js ve Frontend Bağımlılıkları
```bash
# Node.js 16+ gerekli
cd web_interface/frontend
npm install
cd ../..
```

### 5. Arduino Kurulumu
```bash
# Arduino IDE'yi indirin
# arduino/controller.ino dosyasını yükleyin
# Doğru port ve board'u seçin
```

### 6. Model Dosyalarını İndirin
```bash
# YOLO modellerini models/ dizinine yerleştirin
# tabela.pt - Trafik işaretleri modeli
# serit.pt - Şerit algılama modeli
# yolov8n.pt - Genel nesne algılama
# yolov8n-seg.pt - Segmentasyon modeli
```

## 🚀 Kullanım

### Hızlı Başlangıç
```bash
# Tüm sistemi başlatın
python run.py
```

Bu komut otomatik olarak:
- Backend API sunucusunu başlatır (Port 5000)
- Frontend React uygulamasını başlatır (Port 3000)
- Tüm işleme thread'lerini başlatır

### Manuel Başlatma
```bash
# Sadece backend
python main.py

# Sadece web arayüzü
python web_interface/backend/app.py

# Sadece frontend (ayrı terminal)
cd web_interface/frontend
npm start
```

### Erişim Adresleri
- **Web Arayüzü**: http://localhost:3000
- **Backend API**: http://localhost:5000
- **Video Stream**: http://localhost:5000/video_feed
- **API Status**: http://localhost:5000/api/status

## 📁 Proje Yapısı

```
Dursun/
├── 📁 core/                    # Temel algoritmalar ve modeller
│   ├── 📁 algorithms/          # PID, lane fitting, blob detection
│   ├── 📁 controllers/         # Araç kontrol algoritmaları
│   ├── 📁 models/             # Veri modelleri (Lane, TrafficSign, etc.)
│   └── 📁 supervisor/         # Thread yönetimi
├── 📁 modules/                # Ana işleme modülleri
│   ├── 📄 yolo_processor.py   # YOLO nesne algılama
│   ├── 📄 lane_detector.py    # Şerit algılama
│   ├── 📄 depth_analizer.py   # Derinlik analizi
│   ├── 📄 road_processor.py   # Yol verisi işleme
│   ├── 📄 direction_controller.py # Yön kontrolü
│   └── 📄 arduino_cominicator.py # Arduino iletişimi
├── 📁 web_interface/          # Web arayüzü
│   ├── 📁 backend/            # Flask API sunucusu
│   ├── 📁 frontend/           # React SPA
│   ├── 📁 blueprints/         # API endpoint'leri
│   └── 📁 templates/          # HTML şablonları
├── 📁 tests/                  # Test dosyaları
│   ├── 📁 unit/               # Birim testler
│   └── 📁 integration/        # Entegrasyon testleri
├── 📁 arduino/                # Arduino firmware
├── 📁 models/                 # ML model dosyaları
├── 📁 logs/                   # Log dosyaları
├── 📄 config.yaml            # Sistem konfigürasyonu
├── 📄 main.py                # Ana uygulama
├── 📄 run.py                 # Başlatıcı script
├── 📄 orchestator.py         # Süreç yöneticisi
└── 📄 requirements.txt       # Python bağımlılıkları
```

## 🔌 API Dokümantasyonu

### REST Endpoints

#### GET /api/status
Sistem durumu bilgilerini döndürür.

**Response:**
```json
{
  "zed_camera_status": "Connected",
  "arduino_status": "Connected", 
  "detection_results": {
    "traffic_signs": [
      {
        "label": "Stop Sign",
        "confidence": 0.95,
        "bbox": [100, 100, 200, 200]
      }
    ]
  },
  "lane_results": {
    "lanes": [
      {
        "x1": 100, "y1": 200,
        "x2": 300, "y2": 400,
        "type": "bbox"
      }
    ]
  },
  "obstacle_results": {
    "obstacle_detected": false,
    "distance": 2500.0,
    "status": "Clear path"
  },
  "direction_data": {
    "steering_angle": 15,
    "target_speed": 30,
    "vehicle_status": "Düz"
  }
}
```

#### GET /video_feed
MJPEG video stream döndürür.

### WebSocket Events

#### camera_frame
Gerçek zamanlı kamera görüntüsü (Base64 encoded).

#### log_update
Sistem log güncellemeleri.

#### status_update
Periyodik durum güncellemeleri.

## ⚙️ Konfigürasyon

### config.yaml Dosyası
```yaml
# Seri port ayarları
serial:
  port: COM3          # Windows: COM3, Linux: /dev/ttyUSB0
  baud_rate: 9600

# Model dosya yolları
models:
  yolo_traffic_sign: models/tabela.pt
  yolo_lane: models/serit.pt
  yolo_seg: models/yolov8n-seg.pt
  yolo_default: yolov8n.pt

# Kamera ayarları
camera:
  zed_resolution: HD720    # HD720, HD1080, HD2K
  zed_fps: 30
  fallback_webcam_index: 0

# Algılama eşikleri
thresholds:
  depth_obstacle_mm: 2000  # 2 metre
  confidence_threshold: 0.5
  lane_detection_threshold: 0.7

# Loglama
logging:
  level: INFO              # DEBUG, INFO, WARNING, ERROR
  file: logs/dursun.log
  console_level: WARNING

# PID kontrolcü parametreleri
pid:
  steering:
    kp: 1.0
    ki: 0.1
    kd: 0.05
  speed:
    kp: 0.8
    ki: 0.05
    kd: 0.02

# Yol işleme
road:
  default_speed: 30        # km/h
  max_speed: 60
  emergency_brake_distance: 1000  # mm
```

### Ortam Değişkenleri (.env)
```bash
# Backend API
REACT_APP_API_URL=http://localhost:5000
REACT_APP_WS_URL=http://localhost:5000

# Geliştirme modu
DEBUG=True
FLASK_ENV=development

# Kamera ayarları
ZED_CAMERA_ENABLED=True
WEBCAM_INDEX=0

# Arduino
ARDUINO_PORT=COM3
ARDUINO_BAUD=9600
```

## 🧪 Test

### Tüm Testleri Çalıştırma
```bash
# Kapsamlı test suite
pytest --cov=core --cov=modules --cov-report=term-missing

# Sadece birim testler
pytest tests/unit/

# Sadece entegrasyon testleri
pytest tests/integration/

# Belirli bir modül
pytest tests/test_yolo_processor.py -v
```

### Test Kapsamı
- **Birim Testler**: Algoritma ve model testleri
- **Entegrasyon Testleri**: API ve WebSocket testleri
- **Mock Testler**: Donanım bağımlılıkları için
- **Performance Testler**: Bellek ve CPU kullanımı

### CI/CD Pipeline
GitHub Actions ile otomatik test çalıştırma:
```yaml
# .github/workflows/python-package.yml
name: Python package
on: [push, pull_request]
jobs:
  build:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install pytest pytest-cov
    - name: Run tests
      run: pytest --cov=core --cov=modules
```

## 🔧 Sorun Giderme

### Yaygın Sorunlar

#### ZED Kamera Bağlantı Sorunu
```bash
# Kamera durumunu kontrol edin
lsusb | grep ZED

# Sürücü kurulumunu kontrol edin
python -c "import pyzed.sl as sl; print('ZED SDK OK')"

# Webcam fallback test
python -c "import cv2; cap = cv2.VideoCapture(0); print('Webcam OK' if cap.isOpened() else 'Webcam Error')"
```

#### Arduino Bağlantı Sorunu
```bash
# Port listesini kontrol edin
python -m serial.tools.list_ports

# Baud rate uyumluluğunu kontrol edin
# Arduino Serial Monitor'de 9600 baud test edin
```

#### Model Dosyası Bulunamadı
```bash
# Model dosyalarını kontrol edin
ls -la models/

# YOLO modelini indirin
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt -O models/yolov8n.pt
```

#### Port Çakışması
```bash
# Kullanılan portları kontrol edin
netstat -tulpn | grep :5000
netstat -tulpn | grep :3000

# Alternatif port kullanın
export PORT=5001
npm start -- --port 3001
```

### Log Analizi
```bash
# Sistem loglarını izleyin
tail -f logs/dursun.log

# Hata loglarını filtreleyin
grep ERROR logs/dursun.log

# Performans loglarını analiz edin
grep "Processing time" logs/dursun.log
```

### Performans Optimizasyonu
```bash
# GPU kullanımını kontrol edin
nvidia-smi

# CPU kullanımını izleyin
htop

# Bellek kullanımını kontrol edin
free -h
```

## 🤝 Katkıda Bulunma

### Geliştirme Ortamı Kurulumu
```bash
# Geliştirme bağımlılıklarını yükleyin
pip install -r requirements-dev.txt

# Pre-commit hook'larını kurun
pre-commit install

# Code formatting
black .
ruff check .
```

### Katkı Süreci
1. **Fork** edin ve **clone** yapın
2. **Feature branch** oluşturun: `git checkout -b feature/amazing-feature`
3. **Commit** yapın: `git commit -m 'Add amazing feature'`
4. **Push** edin: `git push origin feature/amazing-feature`
5. **Pull Request** açın

### Kod Standartları
- **PEP 8** Python kod stili
- **ESLint** JavaScript/React kod stili
- **Type hints** Python fonksiyonları için
- **Docstrings** tüm public fonksiyonlar için
- **Unit tests** yeni özellikler için

### Issue Raporlama
Hata bildirimi yaparken lütfen şunları ekleyin:
- İşletim sistemi ve Python versiyonu
- Hata mesajının tam metni
- Hatayı yeniden oluşturma adımları
- Beklenen ve gerçek davranış
- Log dosyası çıktısı

## 📄 Lisans

Bu proje MIT Lisansı altında lisanslanmıştır. Detaylar için [LICENSE](LICENSE) dosyasına bakın.

## 🙏 Teşekkürler

- **Ultralytics**: YOLO v8 modeli için
- **Stereolabs**: ZED SDK için
- **OpenCV**: Bilgisayar görüşü kütüphanesi için
- **React**: Modern web arayüzü için
- **Flask**: Backend API framework için

## 📞 İletişim

- **Proje Sahibi**: [GitHub Profili](https://github.com/kullanici)
- **Email**: proje@example.com
- **Discord**: Dursun Community Server
- **Dokümantasyon**: [Wiki Sayfası](https://github.com/kullanici/dursun-project/wiki)

---

<div align="center">

**Dursun Projesi ile Geleceğin Otonom Araçlarını Bugün Geliştirin!**

⭐ Bu projeyi beğendiyseniz yıldız vermeyi unutmayın!

</div>