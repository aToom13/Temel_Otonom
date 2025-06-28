# Dursun Projesi Dosya Yapısı (Yenilenmiş)

```
Dursun/
├── core/
│   ├── algorithms/         # Tüm temel algoritmalar (PID, lane fitting, blob detection, ROI, histogram)
│   ├── controllers/        # Araç ve sistem kontrol algoritmaları (örn. VehicleController)
│   ├── models/             # Veri modelleri (Lane, TrafficSign, Obstacle, VehicleState)
│   └── supervisor/         # Thread ve sistem sağlığı izleme (ThreadSupervisor)
├── modules/                # Yüksek seviye işlevler (YOLO, LaneDetector, DepthAnalyzer, vb.)
├── web_interface/          # Flask backend ve React/JS tabanlı frontend
├── tests/
│   ├── unit/               # Birim testler
│   └── integration/        # Entegrasyon testleri
├── models/                 # ML model dosyaları (.pt)
├── arduino/                # Arduino kodları
├── requirements.txt        # Bağımlılıklar
├── config.yaml             # Konfigürasyon
├── main.py                 # Ana başlatıcı
├── prd.md                  # Proje yol haritası ve analiz
└── README.md               # Genel dokümantasyon
```

> **Not:**
> - Tüm yeni algoritmalar ve veri modelleri `core/` altında toplanmıştır.
> - Thread sağlığı ve izleme için `core/supervisor/thread_supervisor.py` kullanılabilir.
> - Gelişmiş kontrol ve modelleme için `core/algorithms/` ve `core/controllers/` dizinleri kullanılmalıdır.
