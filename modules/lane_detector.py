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

class LaneDetector:
    def __init__(self, model_path=None, config_path=None):
        # Config dosyasını oku
        config_file = config_path or os.path.join(os.path.dirname(__file__), '..', 'config.yaml')
        if os.path.exists(config_file):
            with open(config_file, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            model_conf = config.get('models', {})
            default_model_path = model_conf.get('yolo_lane', 'models/serit.pt')
        else:
            default_model_path = 'models/serit.pt'
        model_path = model_path or default_model_path
        # Check if the model file exists
        if not os.path.exists(model_path):
            logger.warning(f"YOLOv8 segmentation model not found at {model_path}. Please download it. "
                           f"Attempting to download yolov8n-seg.pt...")
            try:
                # Attempt to load from Ultralytics hub, which might download it
                self.model = YOLO('yolov8n-seg.pt')
                logger.info("YOLOv8n-seg model downloaded and loaded successfully.")
            except Exception as e:
                logger.error(f"Error downloading/loading YOLOv8n-seg model: {e}")
                logger.error("LaneDetector initialized without a model. Lane detection will not work.")
                self.model = None
        else:
            self.model = YOLO(model_path)
            logger.info(f"LaneDetector initialized. YOLOv8 segmentation model loaded from {model_path}")

    def detect_lanes(self, frame):
        lane_lines = []
        if self.model is None:
            logger.warning("YOLOv8 segmentation model not loaded. Skipping lane detection.")
            return {"lanes": lane_lines}

        # Perform inference
        results = self.model(frame, verbose=False) # verbose=False to suppress output

        # Process segmentation masks
        for r in results:
            if r.masks is not None:
                for i, mask_data in enumerate(r.masks.data):
                    # Convert mask to numpy array and resize to original frame size
                    mask = mask_data.cpu().numpy()
                    mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
                    mask = (mask > 0.5).astype(np.uint8) * 255 # Binarize mask

                    # Find contours from the mask
                    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                    if contours:
                        # For simplicity, let's just take the largest contour
                        largest_contour = max(contours, key=cv2.contourArea)

                        # Fit a line to the contour (or a polynomial for curves)
                        # Here, we'll just get the bounding box for a simple representation
                        x, y, w, h = cv2.boundingRect(largest_contour)
                        # Or fit a line:
                        # if len(largest_contour) > 1:
                        #     [vx,vy,x,y] = cv2.fitLine(largest_contour, cv2.DIST_L2,0,0.01,0.01)
                        #     # Calculate two points on the line
                        #     lefty = int((-x*vy/vx) + y)
                        #     righty = int(((frame.shape[1]-x)*vy/vx)+y)
                        #     lane_lines.append({"x1": frame.shape[1]-1, "y1": righty, "x2": 0, "y2": lefty})

                        # For now, let's just store the contour points or a simplified representation
                        # A more robust solution would fit polynomials or splines.
                        # Here, we'll store a few points along the contour for visualization/processing
                        points = largest_contour.reshape(-1, 2).tolist()
                        if len(points) > 0:
                            # Store simplified representation (e.g., first and last point, or bounding box)
                            # For now, let's just store the bounding box as a placeholder for a line segment
                            lane_lines.append({"x1": x, "y1": y, "x2": x+w, "y2": y+h, "type": "bbox"})
                            # Or, if you want actual points from the contour:
                            # lane_lines.append({"points": points, "type": "contour"})

        return {"lanes": lane_lines}
