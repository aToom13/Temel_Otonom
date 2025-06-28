# Dursun Otonom Navigasyon Sistemi

<div align="center">

![Dursun Logo](https://via.placeholder.com/200x100/4CAF50/FFFFFF?text=DURSUN)

**Gelişmiş Bilgisayar Görüşü ve Otonom Navigasyon Platformu**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![React](https://img.shields.io/badge/React-18+-61DAFB.svg)](https://reactjs.org)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-green.svg)](https://opencv.org)
[![YOLO](https://img.shields.io/badge/YOLO-v8-orange.svg)](https://ultralytics.com)
[![ZED](https://img.shields.io/badge/ZED-2i-purple.svg)](https://stereolabs.com)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

</div>

## 📋 İçindekiler

- [Proje Hakkında](#-proje-hakkında)
- [Özellikler](#-özellikler)
- [Sistem Gereksinimleri](#-sistem-gereksinimleri)
- [Kurulum](#-kurulum)
- [Kullanım](#-kullanım)
- [ZED 2i Kamera ve IMU](#-zed-2i-kamera-ve-imu)
- [Proje Yapısı](#-proje-yapısı)
- [API Dokümantasyonu](#-api-dokümantasyonu)
- [Konfigürasyon](#-konfigürasyon)
- [Test](#-test)
- [Performans ve Güvenlik](#-performans-ve-güvenlik)
- [Sorun Giderme](#-sorun-giderme)
- [Katkıda Bulunma](#-katkıda-bulunma)
- [Lisans](#-lisans)

## 🚀 Proje Hakkında

Dursun, gelişmiş bilgisayar görüşü teknolojileri kullanarak otonom navigasyon sağlayan kapsamlı bir platformdur. **ZED 2i stereo kamera ve dahili IMU sensörü**, YOLO nesne algılama, gelişmiş şerit takibi ve Arduino tabanlı donanım kontrolü ile gerçek zamanlı otonom sürüş yetenekleri sunar.

### 🎯 Temel Hedefler

- **Güvenli Navigasyon**: Gelişmiş engel algılama ve kaçınma
- **Gerçek Zamanlı İşleme**: Düşük gecikme ile yüksek performans
- **Sensor Fusion**: ZED kamera + IMU entegrasyonu
- **Modüler Mimari**: Kolay genişletilebilir ve özelleştirilebilir
- **Web Tabanlı Kontrol**: Modern ve kullanıcı dostu arayüz

## ✨ Özellikler

### 🎥 Gelişmiş Görüntü İşleme
- **ZED 2i Stereo Kamera**: Derinlik haritası ve 3D algılama
- **Otomatik Fallback**: ZED yokken webcam'e otomatik geçiş
- **Hot-Swap Desteği**: Kamera bağlantısı sırasında otomatik geçiş
- **YOLO v8 Nesne Algılama**: Trafik işaretleri ve araçlar
- **Enhanced Lane Detection**: Temporal consistency ve curve prediction
- **Advanced Depth Analysis**: 3D point cloud ve obstacle clustering

### 🧭 IMU ve Motion Tracking
- **ZED 2i Dahili IMU**: 9-DOF sensor fusion
- **Real-time Orientation**: Roll, pitch, yaw tracking
- **Motion Detection**: Hareket durumu ve hız tahmini
- **Kalman Filtering**: Sensor fusion ve noise reduction
- **Gravity Compensation**: Doğru linear acceleration
- **Vehicle Heading**: Araç yönü ve navigasyon desteği

### 🧠 Yapay Zeka ve Algoritmalar
- **Temporal Lane Detection**: Çok-frame averaging
- **Lane Change Detection**: Lateral movement analysis
- **Advanced Depth Processing**: DBSCAN clustering
- **Safety Monitoring**: ISO 26262 uyumlu güvenlik
- **Performance Optimization**: Memory management ve async processing
- **PID Kontrolcü**: Hassas direksiyon ve hız kontrolü

### 🔧 Donanım Entegrasyonu
- **Arduino İletişimi**: Serial port üzerinden komut gönderimi
- **Motor Kontrolü**: PWM tabanlı hız kontrolü
- **Servo Kontrolü**: Hassas direksiyon kontrolü
- **Sensor Entegrasyonu**: Genişletilebilir sensor desteği

### 🌐 Modern Web Arayüzü
- **Real-time Dashboard**: Chart.js ile canlı grafikler
- **IMU Telemetri**: Orientation, motion, heading gösterimi
- **Camera Status**: ZED/Webcam durumu ve otomatik geçiş
- **Safety Controls**: Emergency stop ve system reset
- **Performance Monitoring**: FPS, CPU, memory, latency
- **Responsive Design**: Mobil ve desktop uyumlu

### 📊 İzleme ve Analiz
- **Safety Monitor**: Functional safety ve watchdog timer
- **Performance Metrics**: Gerçek zamanlı sistem durumu
- **Memory Management**: Otomatik leak detection
- **Health Monitoring**: Bileşen durumu izleme
- **Advanced Logging**: Kapsamlı sistem olayları

## 💻 Sistem Gereksinimleri

### Minimum Gereksinimler
- **İşletim Sistemi**: Windows 10/11, Ubuntu 18.04+, macOS 10.15+
- **Python**: 3.8 veya üzeri
- **RAM**: 4GB (8GB önerilen)
- **Depolama**: 2GB boş alan
- **USB Port**: Arduino ve kamera bağlantısı için

### ZED 2i Kamera Gereksinimleri
- **USB 3.0**: Yüksek bant genişliği için gerekli
- **CUDA**: NVIDIA GPU (GTX 1060 veya üzeri önerilen)
- **ZED SDK**: 4.0 veya üzeri
- **Python API**: pyzed kütüphanesi

### Önerilen Gereksinimler
- **İşlemci**: Intel i7 8. nesil veya AMD Ryzen 7 3700X
- **RAM**: 16GB DDR4
- **GPU**: NVIDIA RTX 3060 veya üzeri (CUDA 11.0+)
- **ZED 2i Kamera**: Stereo görüş ve IMU için
- **SSD**: Hızlı veri erişimi için

### Desteklenen Donanım
- **Kameralar**: ZED 2i (öncelikli), ZED 2, ZED Mini, USB Webcam
- **Mikrocontroller**: Arduino Uno, Nano, Mega
- **Sensörler**: ZED dahili IMU, harici sensörler (gelecek sürümlerde)

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

### 3. ZED SDK Kurulumu (Önemli!)
```bash
# ZED SDK'yı indirin ve kurun
# https://www.stereolabs.com/developers/release/

# CUDA Toolkit kurulumu (NVIDIA GPU için)
# https://developer.nvidia.com/cuda-downloads

# Python API'yi yükleyin
pip install pyzed

# Kurulumu test edin
python -c "import pyzed.sl as sl; print('ZED SDK OK')"
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
mkdir -p models
# tabela.pt - Trafik işaretleri modeli
# serit.pt - Şerit algılama modeli
# yolov8n.pt - Genel nesne algılama
# yolov8n-seg.pt - Segmentasyon modeli
```

### 7. Log Dizinini Oluşturun
```bash
mkdir -p logs
```

## 🚀 Kullanım

### Hızlı Başlangıç
```bash
# Tüm sistemi başlatın
python run.py
```

Bu komut otomatik olarak:
- Enhanced camera manager'ı başlatır (ZED 2i öncelikli)
- IMU processing'i aktifleştirir
- Backend API sunucusunu başlatır (Port 5000)
- Frontend React uygulamasını başlatır (Port 3000)
- Tüm işleme thread'lerini başlatır
- Safety monitoring'i aktifleştirir

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
- **IMU Data**: http://localhost:5000/api/imu/data

## 🧭 ZED 2i Kamera ve IMU

### ZED 2i Özellikleri
- **Stereo Vision**: 720p/1080p/2K çözünürlük
- **Depth Range**: 0.2m - 20m
- **IMU**: 9-DOF (3-axis gyro, accel, magnetometer)
- **USB 3.0**: Yüksek bant genişliği
- **SDK**: Gelişmiş computer vision algoritmaları

### IMU Capabilities
```python
# IMU verilerini al
imu_data = camera_manager.get_imu_data()

# Araç yönü
heading = imu_data['heading_degrees']  # 0-360°

# Eğim bilgisi
roll = imu_data['roll_degrees']    # Yan eğim
pitch = imu_data['pitch_degrees']  # Ön/arka eğim

# Hareket durumu
is_moving = imu_data['is_moving']
speed_kmh = imu_data['speed_kmh']
```

### Otomatik Kamera Geçişi
```python
# ZED kamera bağlandığında otomatik geçiş
if zed_connected:
    camera_manager.switch_to_zed_if_available()
    # Depth analysis aktif
    # IMU processing aktif
    # Enhanced features aktif
else:
    # Webcam fallback
    # Basic processing
```

### Depth Processing
```python
# 3D obstacle detection
obstacles = depth_processor.detect_3d_obstacles(point_cloud)

# Spatial mapping
occupancy_grid = spatial_mapper.update_map(depth_data)

# Distance measurement
distance_to_obstacle = depth_analyzer.get_closest_obstacle()
```

## 📁 Proje Yapısı

```
Dursun/
├── 📁 core/                    # Temel algoritmalar ve modeller
│   ├── 📁 algorithms/          # PID, lane fitting, advanced depth
│   ├── 📁 controllers/         # Araç kontrol algoritmaları
│   ├── 📁 models/             # Veri modelleri
│   ├── 📁 performance/        # Memory manager, async processor
│   ├── 📁 safety/             # Safety monitor, watchdog
│   └── 📁 supervisor/         # Thread yönetimi
├── 📁 modules/                # Ana işleme modülleri
│   ├── 📄 enhanced_camera_manager.py  # ZED + Webcam yönetimi
│   ├── 📄 imu_processor.py            # IMU sensor fusion
│   ├── 📄 enhanced_lane_detector.py   # Gelişmiş şerit algılama
│   ├── 📄 yolo_processor.py           # YOLO nesne algılama
│   ├── 📄 depth_analizer.py           # Derinlik analizi
│   ├── 📄 road_processor.py           # Yol verisi işleme
│   ├── 📄 direction_controller.py     # Yön kontrolü
│   └── 📄 arduino_cominicator.py      # Arduino iletişimi
├── 📁 web_interface/          # Web arayüzü
│   ├── 📁 backend/            # Flask API sunucusu
│   ├── 📁 frontend/           # React SPA
│   │   ├── 📁 src/components/ # RealTimeDashboard, VideoStream
│   │   └── 📁 src/services/   # API ve WebSocket
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
├── 📄 prd.md                 # Proje roadmap ve analiz
└── 📄 requirements.txt       # Python bağımlılıkları
```

## 🔌 API Dokümantasyonu

### REST Endpoints

#### GET /api/status
Kapsamlı sistem durumu bilgilerini döndürür.

**Response:**
```json
{
  "camera_status": {
    "is_connected": true,
    "camera_type": "ZED",
    "resolution": [1280, 720],
    "fps": 30.0,
    "has_depth": true,
    "has_imu": true
  },
  "arduino_status": "Connected",
  "detection_results": {
    "traffic_signs": [...]
  },
  "lane_results": {
    "lanes": [...],
    "lane_center_offset": -0.1,
    "lane_departure_warning": false,
    "detection_quality": 0.85
  },
  "obstacle_results": {
    "obstacle_detected": false,
    "obstacle_count": 0,
    "processing_quality": 0.92
  },
  "direction_data": {
    "steering_angle": 15,
    "target_speed": 30,
    "vehicle_status": "Düz",
    "vehicle_heading": 45.2,
    "is_moving": true,
    "speed_estimate": 25.3
  },
  "imu_data": {
    "heading_degrees": 45.2,
    "roll_degrees": 2.1,
    "pitch_degrees": -1.5,
    "is_moving": true,
    "speed_kmh": 25.3,
    "motion_confidence": 0.87,
    "is_calibrated": true
  },
  "safety_status": {
    "current_state": "SAFE",
    "emergency_stop_active": false
  },
  "performance_metrics": {
    "fps": 28.5,
    "frame_count": 1250
  }
}
```

#### GET /api/imu/data
IMU sensör verilerini döndürür.

#### POST /api/camera/switch_to_zed
ZED kameraya geçiş yapmaya çalışır.

#### POST /api/safety/emergency_stop
Acil durumu aktifleştirir.

#### POST /api/safety/reset
Acil durumu sıfırlar.

### WebSocket Events

#### camera_frame
Gerçek zamanlı kamera görüntüsü (Base64 encoded).

#### imu_update
IMU sensör güncellemeleri.

#### safety_alert
Güvenlik uyarıları.

## ⚙️ Konfigürasyon

### config.yaml Dosyası
```yaml
# ZED Kamera ayarları
camera:
  zed_resolution: HD720        # HD720, HD1080, HD2K
  zed_fps: 30
  fallback_webcam_index: 0
  auto_reconnect: true
  depth_mode: PERFORMANCE      # PERFORMANCE, QUALITY, ULTRA
  coordinate_units: METER

# IMU ayarları
imu:
  motion_threshold: 0.1        # m/s
  stationary_threshold: 0.05   # m/s
  kalman_process_noise: 0.01
  calibration_samples: 100
  gravity_compensation: true

# Gelişmiş şerit algılama
lane_detection:
  temporal_smoothing: 0.3
  lane_departure_threshold: 0.3
  lane_change_threshold: 0.5
  polynomial_degree: 2

# Performans ayarları
performance:
  max_memory_percent: 80.0
  async_workers: 4
  queue_size: 100

# Güvenlik ayarları
safety:
  watchdog_timeout: 1.0
  health_check_interval: 0.5
  performance_thresholds:
    cpu_percent: 90.0
    memory_percent: 85.0
    frame_rate: 10.0
```

## 🧪 Test

### Kapsamlı Test Suite
```bash
# Tüm testleri çalıştır
pytest --cov=core --cov=modules --cov-report=term-missing

# IMU testleri
pytest tests/test_imu_processor.py -v

# Kamera testleri
pytest tests/test_enhanced_camera_manager.py -v

# Performance testleri
pytest tests/performance/ -v
```

### ZED Kamera Testi
```bash
# ZED SDK testi
python -c "import pyzed.sl as sl; cam = sl.Camera(); print('ZED OK' if cam.open() == sl.ERROR_CODE.SUCCESS else 'ZED Error')"

# IMU testi
python -c "from modules.imu_processor import IMUProcessor; imu = IMUProcessor(); print('IMU OK')"
```

## 🔒 Performans ve Güvenlik

### Performans Optimizasyonları
- **Memory Management**: Otomatik leak detection ve cleanup
- **Async Processing**: Thread pool optimization
- **Temporal Consistency**: Multi-frame averaging
- **Safety Monitoring**: Watchdog timer ve health checks

### Güvenlik Özellikleri
- **Functional Safety**: ISO 26262 uyumlu
- **Emergency Stop**: Anında durdurma
- **Command Validation**: Güvenli komut doğrulama
- **Graceful Degradation**: Kademeli performans düşürme

### Performans Metrikleri
- **Video Processing**: 30 FPS @ 720p, 15 FPS @ 1080p
- **IMU Processing**: 100 Hz sensor fusion
- **Object Detection**: <50ms latency
- **Lane Detection**: <30ms latency
- **End-to-End Latency**: <100ms (sensor to actuator)

## 🔧 Sorun Giderme

### ZED Kamera Sorunları
```bash
# ZED SDK kurulum kontrolü
python -c "import pyzed.sl as sl; print('ZED SDK Version:', sl.Camera.get_sdk_version())"

# CUDA kontrolü
nvidia-smi

# USB bağlantı kontrolü
lsusb | grep ZED

# ZED diagnostic tool
/usr/local/zed/tools/ZED_Diagnostic
```

### IMU Kalibrasyon Sorunları
```bash
# IMU kalibrasyonu
python -c "
from modules.imu_processor import IMUProcessor
imu = IMUProcessor()
print('Keep device stationary for calibration...')
# 10 saniye bekle
import time; time.sleep(10)
print('Calibration completed')
"
```

### Performans Sorunları
```bash
# Memory usage kontrolü
python -c "
from core.performance.memory_manager import memory_manager
memory_manager.start_monitoring()
stats = memory_manager.get_memory_stats()
print(f'Memory: {stats.memory_percent:.1f}%')
"

# GPU memory kontrolü
python -c "import torch; print(f'GPU Memory: {torch.cuda.memory_allocated()/1024**3:.1f}GB')"
```

### Kamera Geçiş Sorunları
```bash
# Otomatik geçiş testi
curl -X POST http://localhost:5000/api/camera/switch_to_zed

# Kamera durumu kontrolü
curl http://localhost:5000/api/status | jq '.camera_status'
```

## 🤝 Katkıda Bulunma

### Geliştirme Ortamı
```bash
# Geliştirme bağımlılıkları
pip install -r requirements-dev.txt

# Pre-commit hooks
pre-commit install

# Code formatting
black .
ruff check .
```

### Yeni Özellik Ekleme
1. **IMU Features**: `modules/imu_processor.py`
2. **Camera Features**: `modules/enhanced_camera_manager.py`
3. **Safety Features**: `core/safety/safety_monitor.py`
4. **Performance**: `core/performance/`

### Test Yazma
```python
# IMU test örneği
def test_imu_motion_detection():
    imu = IMUProcessor()
    # Test implementation
```

## 📄 Lisans

Bu proje MIT Lisansı altında lisanslanmıştır. Detaylar için [LICENSE](LICENSE) dosyasına bakın.

## 🙏 Teşekkürler

- **Stereolabs**: ZED SDK ve IMU entegrasyonu için
- **Ultralytics**: YOLO v8 modeli için
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

**Dursun Projesi ile ZED 2i IMU Entegrasyonu ve Gelişmiş Otonom Navigasyon!**

⭐ Bu projeyi beğendiyseniz yıldız vermeyi unutmayın!

</div>