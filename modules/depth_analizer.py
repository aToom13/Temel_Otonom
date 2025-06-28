import cv2
import numpy as np
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

class DepthAnalyzer:
    def __init__(self, config_path=None):
        # Config dosyasını oku
        config_file = config_path or os.path.join(os.path.dirname(__file__), '..', 'config.yaml')
        if os.path.exists(config_file):
            with open(config_file, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            thresholds = config.get('thresholds', {})
            self.threshold_distance = thresholds.get('depth_obstacle_mm', 2000)
        else:
            self.threshold_distance = 2000
        logger.info(f"DepthAnalyzer initialized. Obstacle threshold: {self.threshold_distance} mm")

    def analyze(self, depth_data):
        # Placeholder for depth analysis and obstacle detection
        # 'depth_data' would typically be a depth map from the Zed camera.
        # If Zed camera is not connected, this will receive an RGB frame from webcam.

        if isinstance(depth_data, np.ndarray) and len(depth_data.shape) == 2: # Assuming it's a depth map
            # Simple obstacle detection: check for close objects in a region of interest
            # This is highly simplified. Real depth analysis is more complex.
            height, width = depth_data.shape
            roi_top = int(height * 0.6)
            roi_bottom = int(height * 0.9)
            roi_left = int(width * 0.3)
            roi_right = int(width * 0.7)

            roi_depth = depth_data[roi_top:roi_bottom, roi_left:roi_right]

            # Example: Check if average depth in ROI is below a threshold (e.g., 2 meters)
            # Assuming depth values are in meters or millimeters and need scaling
            # You'll need to calibrate this threshold based on your Zed camera's output
            threshold_distance = self.threshold_distance
            if np.mean(roi_depth) < threshold_distance:
                logger.info(f"Obstacle detected! Distance: {np.mean(roi_depth):.2f} mm")
                return {"obstacle_detected": True, "distance": np.mean(roi_depth), "status": "Obstacle detected based on depth"}
            else:
                logger.info(f"No immediate obstacle. Distance: {np.mean(roi_depth):.2f} mm")
                return {"obstacle_detected": False, "distance": np.mean(roi_depth), "status": "No immediate obstacle"}
        else:
            logger.warning("Depth analysis not possible (No Zed camera depth data)")
            return {"obstacle_detected": False, "distance": "N/A", "status": "Depth analysis not possible (No Zed camera depth data)"}
