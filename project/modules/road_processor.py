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

class RoadProcessor:
    def __init__(self, config_path=None):
        # Config dosyasını oku
        config_file = config_path or os.path.join(os.path.dirname(__file__), '..', 'config.yaml')
        if os.path.exists(config_file):
            with open(config_file, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            self.default_speed = config.get('road', {}).get('default_speed', 30)
        else:
            self.default_speed = 30
        logger.info(f"RoadProcessor initialized. Default speed: {self.default_speed} km/h")

    def process_road(self, combined_data):
        # This module analyzes detected traffic signs, lanes, and obstacles
        # to determine the vehicle's path and status.

        detections = combined_data.get("detections", {})
        lanes = combined_data.get("lanes", {})
        obstacles = combined_data.get("obstacles", {})

        # Placeholder for complex road processing logic
        # This would involve:
        # 1. Fusing data from different sensors/processors.
        # 2. Predicting future path based on lanes.
        # 3. Adjusting path/speed based on traffic signs.
        # 4. Reacting to obstacles (e.g., emergency braking, evasive maneuver).

        current_status = "Düz" # Default status
        target_speed = self.default_speed # Default speed (km/h)
        steering_angle = 0 # Default steering angle (degrees)

        # Example logic based on placeholder data:
        if obstacles.get("obstacle_detected", False):
            current_status = "Dur"
            target_speed = 0
            steering_angle = 0
            logger.warning(f"RoadProcessor: Obstacle detected! Distance: {obstacles.get('distance', 'N/A')}")
        elif detections.get("traffic_signs"):
            for sign in detections["traffic_signs"]:
                if "Stop Sign" in sign["label"]:
                    current_status = "Dur"
                    target_speed = 0
                    logger.info("RoadProcessor: Stop sign detected!")
                elif "Speed Limit 60" in sign["label"]:
                    target_speed = min(target_speed, 60) # Adjust speed if needed
                    logger.info("RoadProcessor: Speed limit 60 detected!")

        # Basic lane following (very simplified)
        if lanes.get("lanes"):
            # In a real scenario, you'd calculate a center line and deviation
            # For this placeholder, let's assume a simple steering adjustment
            # based on the presence of lanes.
            # This is highly dependent on how lane_detector returns data.
            pass # Add actual lane processing here

        # Return processed data for direction controller
        return {
            "steering_angle": steering_angle,
            "target_speed": target_speed,
            "vehicle_status": current_status,
            "timestamp": combined_data.get("timestamp", "N/A") # Add timestamp for real-time data
        }
