import threading
import time
import cv2
import numpy as np
import logging
import os
import yaml

from modules.yolo_processor import YoloProcessor
from modules.lane_detector import LaneDetector
from modules.depth_analizer import DepthAnalyzer
from modules.road_processor import RoadProcessor
from modules.direction_controller import DirectionController
from modules.arduino_cominicator import ArduinoCommunicator

# Logging ayarları
config_file = os.path.join(os.path.dirname(__file__), 'config.yaml')
if os.path.exists(config_file):
    with open(config_file, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    log_conf = config.get('logging', {})
    log_level = getattr(logging, log_conf.get('level', 'INFO').upper(), logging.INFO)
    log_file = log_conf.get('file', 'dursun.log')
else:
    log_level = logging.INFO
    log_file = 'dursun.log'
# Sadece uyarı ve hata mesajlarını göster, info ve debug'u bastır
logging.basicConfig(
    level=log_conf.get('console_level', 'WARNING').upper() if 'console_level' in log_conf else logging.WARNING,
    format='%(asctime)s [%(levelname)s] %(threadName)s: %(message)s',
    handlers=[logging.FileHandler(log_file, encoding='utf-8'), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Global variables for sharing data between threads
camera_frame = None
camera_frame_lock = threading.Lock()
processed_frame = None
processed_frame_lock = threading.Lock()
detection_results = {}
detection_results_lock = threading.Lock()
lane_results = {}
lane_results_lock = threading.Lock()
obstacle_results = {}
obstacle_results_lock = threading.Lock()
direction_data = {}
direction_data_lock = threading.Lock()
arduino_status = "Disconnected"
arduino_status_lock = threading.Lock()
zed_camera_status = "Disconnected"
zed_camera_status_lock = threading.Lock()

# --- Camera Stream (Placeholder for Zed2i or Webcam) ---
def camera_stream_thread():
    global camera_frame, zed_camera_status
    try:
        import pyzed.sl as sl
        zed = sl.Camera()
        init_params = sl.InitParameters()
        init_params.camera_resolution = sl.RESOLUTION.HD720
        init_params.camera_fps = 30
        err = zed.open(init_params)
        if err != sl.ERROR_CODE.SUCCESS:
            logger.error(f"ZED Camera Error: {err}. Falling back to webcam.")
            with zed_camera_status_lock:
                zed_camera_status = "Error: " + str(err)
            cap = cv2.VideoCapture(0) # Fallback to webcam
            if not cap.isOpened():
                logger.error("Error: Could not open webcam.")
                with zed_camera_status_lock:
                    zed_camera_status = "Error: Webcam not found."
                return
            else:
                with zed_camera_status_lock:
                    zed_camera_status = "Webcam Active (ZED not found)"
        else:
            logger.info("ZED Camera Opened Successfully.")
            with zed_camera_status_lock:
                zed_camera_status = "Connected"
            runtime_parameters = sl.RuntimeParameters()
            image = sl.Mat()
            depth = sl.Mat()
            while True:
                if zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
                    zed.retrieve_image(image, sl.VIEW.LEFT)
                    zed.retrieve_measure(depth, sl.MEASURE.DEPTH)
                    with camera_frame_lock:
                        camera_frame = image.get_data()
                time.sleep(0.01)
    except ImportError:
        logger.warning("pyzed not found. Using webcam as fallback.")
        with zed_camera_status_lock:
            zed_camera_status = "pyzed not found. Using Webcam."
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            logger.error("Error: Could not open webcam.")
            with zed_camera_status_lock:
                zed_camera_status = "Error: Webcam not found."
            return
        else:
            with zed_camera_status_lock:
                zed_camera_status = "Webcam Active (pyzed not found)"
    with zed_camera_status_lock:
        zed_status = zed_camera_status
    if zed_status != "Connected":
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.error("Failed to grab frame from webcam.")
                break
            with camera_frame_lock:
                camera_frame = frame
            time.sleep(0.01)
    if 'zed' in locals() and zed_status == "Connected":
        zed.close()
    elif 'cap' in locals():
        cap.release()

# --- Processing Threads ---
def yolo_processor_thread():
    global camera_frame, processed_frame, detection_results
    yolo_processor = YoloProcessor()
    while True:
        with camera_frame_lock:
            frame = camera_frame.copy() if camera_frame is not None else None
        if frame is not None:
            proc_frame, det_results = yolo_processor.process_frame(frame)
            with processed_frame_lock:
                processed_frame = proc_frame
            with detection_results_lock:
                detection_results = det_results
        time.sleep(0.1) # Process every 100ms

def lane_detector_thread():
    global camera_frame, lane_results
    lane_detector = LaneDetector()
    while True:
        with camera_frame_lock:
            frame = camera_frame.copy() if camera_frame is not None else None
        if frame is not None:
            lane_res = lane_detector.detect_lanes(frame)
            with lane_results_lock:
                lane_results = lane_res
        time.sleep(0.1)

def depth_analyzer_thread():
    global camera_frame, obstacle_results, zed_camera_status
    depth_analyzer = DepthAnalyzer()
    while True:
        with camera_frame_lock:
            frame = camera_frame.copy() if camera_frame is not None else None
        with zed_camera_status_lock:
            zed_status = zed_camera_status
        if frame is not None and zed_status == "Connected": # Only run if Zed is connected
            obs_res = depth_analyzer.analyze(frame) # Placeholder: analyze RGB for now
            with obstacle_results_lock:
                obstacle_results = obs_res
        elif zed_status != "Connected":
            with obstacle_results_lock:
                obstacle_results = {"status": "Depth analysis unavailable (ZED camera not connected or pyzed not found)"}
        time.sleep(0.1)

def road_processor_thread():
    global detection_results, lane_results, obstacle_results, direction_data
    road_processor = RoadProcessor()
    while True:
        with detection_results_lock:
            det = detection_results.copy()
        with lane_results_lock:
            lanes = lane_results.copy()
        with obstacle_results_lock:
            obs = obstacle_results.copy()
        combined_data = {
            "detections": det,
            "lanes": lanes,
            "obstacles": obs
        }
        dir_data = road_processor.process_road(combined_data)
        with direction_data_lock:
            direction_data = dir_data
        time.sleep(0.1)

def direction_controller_thread():
    global direction_data
    direction_controller = DirectionController()
    while True:
        with direction_data_lock:
            dir_data = direction_data.copy() if direction_data else None
        if dir_data:
            direction_controller.control(dir_data)
        time.sleep(0.1)

def arduino_communicator_thread():
    global direction_data, arduino_status
    arduino_communicator = ArduinoCommunicator()
    while True:
        try:
            if arduino_communicator.is_connected():
                with arduino_status_lock:
                    arduino_status = "Connected"
                with direction_data_lock:
                    dir_data = direction_data.copy() if direction_data else None
                if dir_data:
                    arduino_communicator.send_data(dir_data)
            else:
                with arduino_status_lock:
                    arduino_status = "Disconnected"
                arduino_communicator.connect() # Try to reconnect
        except Exception as e:
            with arduino_status_lock:
                arduino_status = f"Error: {e}"
            logger.error(f"Arduino Communication Error: {e}")
        time.sleep(1) # Try to send data/reconnect every second

def start_processing_threads():
    # Start all processing threads
    threading.Thread(target=camera_stream_thread, daemon=True).start()
    threading.Thread(target=yolo_processor_thread, daemon=True).start()
    threading.Thread(target=lane_detector_thread, daemon=True).start()
    threading.Thread(target=depth_analyzer_thread, daemon=True).start()
    threading.Thread(target=road_processor_thread, daemon=True).start()
    threading.Thread(target=direction_controller_thread, daemon=True).start()
    threading.Thread(target=arduino_communicator_thread, daemon=True).start()

def main():
    start_processing_threads()
    logger.info("Tüm iş parçacıkları başlatıldı. Çıkmak için Ctrl+C.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Çıkılıyor...")

if __name__ == "__main__":
    main()
