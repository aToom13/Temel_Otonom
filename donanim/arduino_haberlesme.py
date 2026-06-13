import time


class ArduinoBaglanti:
    """
    Arduino mikrodenetleyicisi ile USB UART seri port üzerinden
    hız ve direksiyon komutlarını ileten haberleşme sınıfı.
    Hem 'ascii_ha' (H<hiz>A<aci>\n) hem de 'kat4_rv' (r015v110) protokollerini destekler.
    Bağlantı yoksa veya simulasyon=True ise simülasyon modunda güvenle çalışır.
    """

    def __init__(self, port="/dev/ttyUSB0", baudrate=115200, aktif=False, simulasyon=None, protokol="ascii_ha"):
        self.port = port
        self.baudrate = baudrate
        self.protokol = protokol  # 'ascii_ha' veya 'kat4_rv'

        # simulasyon parametresi verilmişse aktiflik durumunu belirle
        if simulasyon is not None:
            self.aktif = not simulasyon
        else:
            self.aktif = aktif

        self.seri = None
        self.simulasyon_modu = not self.aktif
        self.son_komut = ""

        if self.aktif:
            self.baglan()

    def baglan(self):
        """Arduino seri port bağlantısını başlatır."""
        try:
            import serial
            self.seri = serial.Serial(self.port, self.baudrate, timeout=0.1)
            time.sleep(1.5)  # Arduino bootloader reset bekleme süresi
            self.simulasyon_modu = False
            print(f"[BILGI] Arduino bağlantısı kuruldu: {self.port} @ {self.baudrate}")
        except Exception as e:
            print(f"[UYARI] Arduino portuna bağlanılamadı ({self.port}): {e}")
            print("[BILGI] Simülasyon moduna geçildi (seri çıkış yapılmayacak).")
            self.simulasyon_modu = True
            self.seri = None

    def komut_olustur(self, hiz, aci, protokol=None):
        """
        Belirtilen protokole göre bayt dizisini hazırlar.
        - 'ascii_ha': H110A-12.5\n
        - 'kat4_rv': r012v110 (Kat4 Teknofest Robotaksi protokolü)
        """
        proto = protokol or self.protokol
        pwm_hiz = int(max(0, min(255, hiz)))

        if proto == "kat4_rv":
            # Kat4 protokolü: r/l/f + 3 basamak adim/aci + v + 3 basamak hiz
            if abs(aci) < 1.0:
                yon = 'f'
                val = 0
            elif aci > 0:
                yon = 'r'
                val = int(abs(aci))
            else:
                yon = 'l'
                val = int(abs(aci))
            return f"{yon}{val:03d}v{pwm_hiz:03d}\n"
        else:
            # Standart ASCII H<hiz>A<aci>\n protokolü
            return f"H{pwm_hiz}A{aci:.1f}\n"

    def komut_gonder(self, hiz, aci):
        """Oluşturulan komutu Arduino'ya gönderir veya simülasyonda kaydeder."""
        veri = self.komut_olustur(hiz, aci)
        self.son_komut = veri.strip()

        if self.simulasyon_modu or self.seri is None:
            return self.son_komut

        try:
            self.seri.write(veri.encode('ascii'))
            return self.son_komut
        except Exception as e:
            print(f"[HATA] Seri port yazma hatası: {e}")
            self.simulasyon_modu = True
            return self.son_komut

    def kapat(self):
        """Bağlantıyı güvenle sonlandırır ve aracı durdurur."""
        if self.seri and self.seri.is_open:
            try:
                durdurma = self.komut_olustur(0, 0.0)
                self.seri.write(durdurma.encode('ascii'))
                self.seri.close()
            except Exception:
                pass


ArduinoHaberlesme = ArduinoBaglanti
