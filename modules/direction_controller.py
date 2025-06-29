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

class DirectionController:
    def __init__(self, config_path=None):
        # Config dosyasını oku (ileride PID parametreleri eklenebilir)
        config_file = config_path or os.path.join(os.path.dirname(__file__), '..', 'config.yaml')
        if os.path.exists(config_file):
            with open(config_file, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            self.pid_params = config.get('pid', {})
        else:
            self.pid_params = {}
        logger.info("DirectionController initialized.")

    def control(self, direction_data):
        # This module takes the processed road data and determines
        # the final control signals for the vehicle (steering, speed, status).

        steering_angle = direction_data.get("steering_angle", 0)
        target_speed = direction_data.get("target_speed", 0)
        vehicle_status = direction_data.get("vehicle_status", "Düz")

        logger.info(f"DirectionController: Steering: {steering_angle} deg, Speed: {target_speed} km/h, Status: {vehicle_status}")

        # You would typically return these values or store them for Arduino communication
        return {
            "angle": steering_angle,
            "speed": target_speed,
            "status": vehicle_status
        }
