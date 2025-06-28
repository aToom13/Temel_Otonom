# Temel Projesi

## Proje Hakkında

Temel Projesi, bilgisayar görüşü, donanım kontrolü ve web tabanlı bir arayüzü bir araya getiren otonom bir navigasyon veya robotik sistemdir. Bu proje, bir aracın veya mobil robotun çevresini algılamasına, yolunu planlamasına ve donanım bileşenlerini kontrol etmesine olanak tanır.

**Ana Özellikler:**

*   **Bilgisayar Görüşü:** YOLO modelleri kullanarak nesne algılama ve segmentasyon, şerit algılama ve derinlik analizi ile çevresel farkındalık.
*   **Navigasyon ve Kontrol:** Yol verilerini işleme, aracın yönünü hassas bir şekilde kontrol etme ve Arduino tabanlı donanım ile sorunsuz iletişim.
*   **Web Arayüzü:** Sistemin durumunu gerçek zamanlı olarak izlemek, kontrol etmek ve görselleştirmek için kullanıcı dostu bir web uygulaması.

## Kurulum

Projeyi yerel makinenizde kurmak ve çalıştırmak için aşağıdaki adımları izleyin.

### Ön Koşullar

*   Python 3.8 veya üzeri
*   pip (Python paket yöneticisi)
*   Arduino IDE (Arduino kodunu yüklemek için)
*   Gerekli donanım (Arduino kartı, sensörler, motorlar vb.)

### Adımlar

1.  **Depoyu Klonlayın:**
    ```bash
    git clone <depo_adresiniz>
    cd Temel
    ```

2.  **Python Bağımlılıklarını Yükleyin:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Arduino Kurulumu:**
    *   `arduino/controller.ino` dosyasını Arduino IDE'de açın.
    *   Arduino kartınızı bilgisayarınıza bağlayın.
    *   Doğru kartı ve portu seçtiğinizden emin olun.
    *   Kodu Arduino kartınıza yükleyin.

4.  **Model Dosyaları:**
    `models/` dizininde gerekli tüm `.pt` model dosyalarının (örn. `serit.pt`, `tabela.pt`, `yolov8n-seg.pt`, `yolov8n.pt`) bulunduğundan emin olun. Bu modeller genellikle önceden eğitilmiş olarak sağlanır veya özel veri kümeleriyle eğitilir.

## Kullanım

### Ana Uygulamayı Çalıştırma

Projenin ana Python uygulamasını başlatmak için:

```bash
python main.py
```
Bu komut, bilgisayar görüşü ve kontrol mantığını başlatacaktır.

### Web Arayüzünü Çalıştırma

Web arayüzü, sistemin durumunu izlemek ve kontrol etmek için kullanılır.

1.  **Web Arayüzü Dizinine Gidin:**
    ```bash
    cd web_interface
    ```

2.  **Flask Uygulamasını Başlatın:**
    ```bash
    python app.py
    ```
    Uygulama varsayılan olarak `http://127.0.0.1:5000` adresinde çalışacaktır. Web tarayıcınızdan bu adrese giderek arayüze erişebilirsiniz.

## Proje Yapısı

*   `main.py`: Projenin ana yürütülebilir dosyası.
*   `requirements.txt`: Python bağımlılıklarını listeler.
*   `arduino/`: Arduino kartına yüklenecek bellenim kodunu içerir.
*   `models/`: Makine öğrenimi modellerini (`.pt` dosyaları) barındırır.
*   `modules/`: Projenin temel mantığını içeren Python modülleri (örn. `lane_detector.py`, `yolo_processor.py`).
*   `tests/`: Proje için birim ve entegrasyon testlerini içerir.
*   `web_interface/`: Web arayüzü uygulamasını (Flask backend, HTML/CSS/JS frontend) içerir.

## Katkıda Bulunma

Katkılarınız memnuniyetle karşılanır! Lütfen bir özellik eklemek veya bir hatayı düzeltmek için bir çekme isteği (pull request) göndermeden önce mevcut kodlama standartlarına uyun.

## Lisans

Bu proje için bir lisans dosyası (`LICENSE`) yakında eklenecektir.
