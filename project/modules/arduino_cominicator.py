import serial
import time
import os
import yaml
import logging

# Logging ayarları
config_file = os.path.join(os.path.dirname(__file__), '..', 'config.yaml')
if os.path.exists(config_file):
    with open(config_file, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    log_conf = config.get('logging', {})
    log_level = getattr(logging, log_conf.get('level', 'INFO').upper(), logging.INFO)
    log_file = log_conf.get('file', 'dursun.log')
else:
    log_level = logging.INFO
    log_file = 'dursun.log'
logging.basicConfig(
    level=log_level,
    format='%(asctime)s [%(levelname)s] %(threadName)s: %(message)s',
    handlers=[logging.FileHandler(log_file, encoding='utf-8'), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

class ArduinoCommunicator:
    def __init__(self, port=None, baud_rate=None, config_path=None):
        # Config dosyasını oku
        config_file = config_path or os.path.join(os.path.dirname(__file__), '..', 'config.yaml')
        if os.path.exists(config_file):
            with open(config_file, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            serial_conf = config.get('serial', {})
            default_port = serial_conf.get('port', 'COM3')
            default_baud = serial_conf.get('baud_rate', 9600)
        else:
            default_port = 'COM3'
            default_baud = 9600
        self.port = port or default_port
        self.baud_rate = baud_rate or default_baud
        self.serial_connection = None
        self.connect()
        logger.info(f"ArduinoCommunicator initialized for port {self.port} at {self.baud_rate} baud.")

    def connect(self):
        if self.serial_connection and self.serial_connection.is_open:
            logger.info("Arduino already connected.")
            return True
        try:
            self.serial_connection = serial.Serial(self.port, self.baud_rate, timeout=1)
            time.sleep(2) # Give Arduino time to reset
            logger.info(f"Successfully connected to Arduino on {self.port}")
            return True
        except serial.SerialException as e:
            logger.error(f"Could not connect to Arduino on {self.port}: {e}")
            self.serial_connection = None
            return False

    def is_connected(self):
        return self.serial_connection is not None and self.serial_connection.is_open

    def send_data(self, data):
        if not self.is_connected():
            logger.warning("Arduino not connected. Cannot send data.")
            return False
        try:
            angle = int(data.get("angle", 0))
            speed = int(data.get("speed", 0))
            status = data.get("status", "Düz")
            status_code = 0
            if status == "Düz":
                status_code = 0
            elif status == "Viraj":
                status_code = 1
            elif status == "Dur":
                status_code = 2
            message = f"A{angle}S{speed}V{status_code}\n"
            self.serial_connection.write(message.encode())
            logger.info(f"Sent to Arduino: {message.strip()}")
            return True
        except Exception as e:
            logger.error(f"Error sending data to Arduino: {e}")
            self.serial_connection.close()
            self.serial_connection = None
            return False

    def read_data(self):
        if not self.is_connected():
            return None
        try:
            if self.serial_connection.in_waiting > 0:
                line = self.serial_connection.readline().decode('utf-8').strip()
                logger.info(f"Received from Arduino: {line}")
                return line
        except Exception as e:
            logger.error(f"Error reading data from Arduino: {e}")
            self.serial_connection.close()
            self.serial_connection = None
            return None
        return None

    def disconnect(self):
        if self.serial_connection and self.serial_connection.is_open:
            self.serial_connection.close()
            logger.info("Disconnected from Arduino.")
        self.serial_connection = None
