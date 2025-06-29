import cv2
import numpy as np
from ultralytics import YOLO
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

class YoloProcessor:
    def __init__(self, model_path=None, config_path=None):
        # Config dosyasını oku
        config_file = config_path or os.path.join(os.path.dirname(__file__), '..', 'config.yaml')
        if os.path.exists(config_file):
            with open(config_file, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            model_conf = config.get('models', {})
            default_model_path = model_conf.get('yolo_traffic_sign', 'models/tabela.pt')
        else:
            default_model_path = 'models/tabela.pt'
        model_path = model_path or default_model_path
        # Check if the model file exists
        if not os.path.exists(model_path):
            logger.warning(f"YOLOv8 model not found at {model_path}. Please download it. "
                           f"Attempting to download yolov8n.pt...")
            try:
                # Attempt to load from Ultralytics hub, which might download it
                self.model = YOLO('yolov8n.pt')
                logger.info("YOLOv8n model downloaded and loaded successfully.")
            except Exception as e:
                logger.error(f"Error downloading/loading YOLOv8n model: {e}")
                logger.error("YoloProcessor initialized without a model. Detection will not work.")
                self.model = None
        else:
            self.model = YOLO(model_path)
            logger.info(f"YoloProcessor initialized. YOLOv8 model loaded from {model_path}")

    def process_frame(self, frame):
        detection_results = {"traffic_signs": []}
        display_frame = frame.copy()

        if self.model is None:
            logger.warning("YOLOv8 model not loaded. Skipping detection.")
            return display_frame, detection_results

        # Perform inference
        results = self.model(frame, verbose=False) # verbose=False to suppress output

        # Process results
        for r in results:
            boxes = r.boxes  # Boxes object for bbox outputs
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = float(box.conf[0])
                class_id = int(box.cls[0])
                label = self.model.names[class_id]

                # You might want to filter for specific traffic sign classes here
                # For example, if you have a custom trained model for traffic signs
                # if label in ["stop_sign", "speed_limit"]: # Example filtering

                detection_results["traffic_signs"].append({
                    "label": label,
                    "confidence": confidence,
                    "bbox": [x1, y1, x2, y2]
                })

                # Draw bounding box and label on the frame
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(display_frame, f"{label} ({confidence:.2f})", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        return display_frame, detection_results