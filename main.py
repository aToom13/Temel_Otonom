import threading
import time
import cv2
import numpy as np
import logging
import os
import yaml

# Import enhanced modules
from modules.yolo_processor import YoloProcessor
from modules.enhanced_lane_detector import EnhancedLaneDetector
from modules.depth_analizer import DepthAnalyzer
from modules.road_processor import RoadProcessor
from modules.direction_controller import DirectionController
from modules.arduino_cominicator import ArduinoCommunicator

# Import performance and safety modules
from core.performance.memory_manager import memory_manager
from core.performance.async_processor import async_processor, ProcessingTask
from core.safety.safety_monitor import safety_monitor, SystemComponent, HealthStatus, SafetyState

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

# Enhanced logging configuration
logging.basicConfig(
    level=log_conf.get('console_level', 'WARNING').upper() if 'console_level' in log_conf else logging.WARNING,
    format='%(asctime)s [%(levelname)s] %(threadName)s: %(message)s',
    handlers=[
        logging.FileHandler(log_file, encoding='utf-8'),
        logging.StreamHandler()
    ]
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

# Performance metrics
performance_metrics = {
    'frame_count': 0,
    'processing_times': [],
    'fps': 0.0,
    'last_fps_update': time.time()
}
performance_metrics_lock = threading.Lock()

def update_performance_metrics(processing_time: float):
    """Performans metriklerini güncelle"""
    with performance_metrics_lock:
        performance_metrics['frame_count'] += 1
        performance_metrics['processing_times'].append(processing_time)
        
        # Keep only last 30 processing times
        if len(performance_metrics['processing_times']) > 30:
            performance_metrics['processing_times'] = performance_metrics['processing_times'][-30:]
        
        # Update FPS every second
        current_time = time.time()
        if current_time - performance_metrics['last_fps_update'] >= 1.0:
            performance_metrics['fps'] = len(performance_metrics['processing_times']) / (
                current_time - performance_metrics['last_fps_update']
            )
            performance_metrics['last_fps_update'] = current_time

def create_camera_health_callback():
    """Kamera sağlık kontrolü callback'i oluştur"""
    def check_camera_health():
        with zed_camera_status_lock:
            status = zed_camera_status
        
        if "Connected" in status:
            return HealthStatus(
                component=SystemComponent.CAMERA,
                status=SafetyState.SAFE,
                message="Camera operating normally",
                timestamp=time.time(),
                metrics={"status": status}
            )
        else:
            return HealthStatus(
                component=SystemComponent.CAMERA,
                status=SafetyState.WARNING if "Webcam" in status else SafetyState.CRITICAL,
                message=f"Camera issue: {status}",
                timestamp=time.time(),
                metrics={"status": status}
            )
    
    return check_camera_health

def create_arduino_health_callback():
    """Arduino sağlık kontrolü callback'i oluştur"""
    def check_arduino_health():
        with arduino_status_lock:
            status = arduino_status
        
        if status == "Connected":
            return HealthStatus(
                component=SystemComponent.ARDUINO,
                status=SafetyState.SAFE,
                message="Arduino communication normal",
                timestamp=time.time(),
                metrics={"status": status}
            )
        else:
            return HealthStatus(
                component=SystemComponent.ARDUINO,
                status=SafetyState.CRITICAL,
                message=f"Arduino issue: {status}",
                timestamp=time.time(),
                metrics={"status": status}
            )
    
    return check_arduino_health

def create_processing_health_callback():
    """İşleme sağlık kontrolü callback'i oluştur"""
    def check_processing_health():
        with performance_metrics_lock:
            fps = performance_metrics['fps']
            avg_processing_time = np.mean(performance_metrics['processing_times']) if performance_metrics['processing_times'] else 0
        
        if fps > 15 and avg_processing_time < 0.1:
            status = SafetyState.SAFE
            message = "Processing performance good"
        elif fps > 10 and avg_processing_time < 0.2:
            status = SafetyState.WARNING
            message = "Processing performance degraded"
        else:
            status = SafetyState.CRITICAL
            message = "Processing performance critical"
        
        return HealthStatus(
            component=SystemComponent.PROCESSING,
            status=status,
            message=message,
            timestamp=time.time(),
            metrics={"fps": fps, "avg_processing_time": avg_processing_time}
        )
    
    return check_processing_health

# --- Enhanced Camera Stream ---
def camera_stream_thread():
    global camera_frame, zed_camera_status
    
    camera = None
    zed = None
    
    try:
        # Try ZED camera first
        import pyzed.sl as sl
        zed = sl.Camera()
        init_params = sl.InitParameters()
        
        # Load camera config
        if os.path.exists(config_file):
            camera_config = config.get('camera', {})
            resolution_str = camera_config.get('zed_resolution', 'HD720')
            fps = camera_config.get('zed_fps', 30)
            
            # Set resolution
            if resolution_str == 'HD720':
                init_params.camera_resolution = sl.RESOLUTION.HD720
            elif resolution_str == 'HD1080':
                init_params.camera_resolution = sl.RESOLUTION.HD1080
            elif resolution_str == 'HD2K':
                init_params.camera_resolution = sl.RESOLUTION.HD2K
            
            init_params.camera_fps = fps
        else:
            init_params.camera_resolution = sl.RESOLUTION.HD720
            init_params.camera_fps = 30
        
        err = zed.open(init_params)
        if err != sl.ERROR_CODE.SUCCESS:
            logger.error(f"ZED Camera Error: {err}. Falling back to webcam.")
            with zed_camera_status_lock:
                zed_camera_status = f"Error: {err}"
            zed = None
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
                        camera_frame = {
                            'rgb': image.get_data(),
                            'depth': depth.get_data()
                        }
                else:
                    logger.warning("Failed to grab ZED frame")
                
                time.sleep(0.01)
                
    except ImportError:
        logger.warning("pyzed not found. Using webcam as fallback.")
        with zed_camera_status_lock:
            zed_camera_status = "pyzed not found. Using Webcam."
    except Exception as e:
        logger.error(f"ZED camera initialization failed: {e}")
        with zed_camera_status_lock:
            zed_camera_status = f"ZED Error: {e}"
    
    # Fallback to webcam
    if zed is None:
        try:
            webcam_index = 0
            if os.path.exists(config_file):
                webcam_index = config.get('camera', {}).get('fallback_webcam_index', 0)
            
            camera = cv2.VideoCapture(webcam_index)
            if not camera.isOpened():
                logger.error("Error: Could not open webcam.")
                with zed_camera_status_lock:
                    zed_camera_status = "Error: Webcam not found."
                return
            else:
                with zed_camera_status_lock:
                    zed_camera_status = "Webcam Active (ZED not available)"
                
                logger.info(f"Webcam opened on index {webcam_index}")
                
                while True:
                    ret, frame = camera.read()
                    if not ret:
                        logger.error("Failed to grab frame from webcam.")
                        break
                    
                    with camera_frame_lock:
                        camera_frame = {
                            'rgb': frame,
                            'depth': None
                        }
                    
                    time.sleep(0.01)
                    
        except Exception as e:
            logger.error(f"Webcam initialization failed: {e}")
            with zed_camera_status_lock:
                zed_camera_status = f"Webcam Error: {e}"
    
    # Cleanup
    if zed:
        zed.close()
    if camera:
        camera.release()

# --- Enhanced Processing Threads ---
def yolo_processor_thread():
    global camera_frame, processed_frame, detection_results
    
    yolo_processor = YoloProcessor()
    
    while True:
        start_time = time.time()
        
        with camera_frame_lock:
            frame_data = camera_frame.copy() if camera_frame is not None else None
        
        if frame_data is not None and frame_data['rgb'] is not None:
            try:
                # Create processing task
                task = ProcessingTask(
                    id=f"yolo_{int(time.time() * 1000)}",
                    data=frame_data['rgb'],
                    processor=yolo_processor.process_frame,
                    priority=2  # High priority
                )
                
                # Submit to async processor
                if async_processor.submit_task(task):
                    # For now, process synchronously
                    proc_frame, det_results = yolo_processor.process_frame(frame_data['rgb'])
                    
                    with processed_frame_lock:
                        processed_frame = proc_frame
                    with detection_results_lock:
                        detection_results = det_results
                    
                    processing_time = time.time() - start_time
                    update_performance_metrics(processing_time)
                    
            except Exception as e:
                logger.error(f"YOLO processing failed: {e}")
        
        time.sleep(0.1)

def lane_detector_thread():
    global camera_frame, lane_results
    
    lane_detector = EnhancedLaneDetector()
    
    while True:
        start_time = time.time()
        
        with camera_frame_lock:
            frame_data = camera_frame.copy() if camera_frame is not None else None
        
        if frame_data is not None and frame_data['rgb'] is not None:
            try:
                lane_res = lane_detector.detect_lanes_with_tracking(frame_data['rgb'])
                
                with lane_results_lock:
                    lane_results = {
                        'lanes': [
                            {
                                'lane_type': lane.lane_type,
                                'confidence': lane.confidence,
                                'curvature': lane.curvature,
                                'points_count': len(lane.points)
                            }
                            for lane in lane_res.lanes
                        ],
                        'lane_center_offset': lane_res.lane_center_offset,
                        'lane_departure_warning': lane_res.lane_departure_warning,
                        'lane_change_detected': lane_res.lane_change_detected,
                        'road_curvature': lane_res.road_curvature,
                        'detection_quality': lane_res.detection_quality
                    }
                
                processing_time = time.time() - start_time
                logger.debug(f"Lane detection completed in {processing_time:.3f}s")
                
            except Exception as e:
                logger.error(f"Lane detection failed: {e}")
                with lane_results_lock:
                    lane_results = {'lanes': [], 'detection_quality': 0.0}
        
        time.sleep(0.1)

def depth_analyzer_thread():
    global camera_frame, obstacle_results, zed_camera_status
    
    # Import advanced depth processor
    from core.algorithms.advanced_depth_processing import AdvancedDepthProcessor
    depth_analyzer = AdvancedDepthProcessor()
    
    while True:
        start_time = time.time()
        
        with camera_frame_lock:
            frame_data = camera_frame.copy() if camera_frame is not None else None
        with zed_camera_status_lock:
            zed_status = zed_camera_status
        
        if frame_data is not None:
            try:
                if "Connected" in zed_status and frame_data.get('depth') is not None:
                    # Use advanced depth processing
                    depth_result = depth_analyzer.process_depth_map(frame_data['depth'])
                    
                    with obstacle_results_lock:
                        obstacle_results = {
                            'obstacle_detected': depth_result['obstacle_count'] > 0,
                            'obstacle_count': depth_result['obstacle_count'],
                            'obstacles': [
                                {
                                    'center': {'x': obs.center.x, 'y': obs.center.y, 'z': obs.center.z},
                                    'size': obs.size,
                                    'confidence': obs.confidence
                                }
                                for obs in depth_result['obstacles']
                            ],
                            'processing_quality': depth_result['processing_quality'],
                            'status': "Advanced depth analysis active"
                        }
                else:
                    # Fallback to basic analysis
                    basic_result = self._basic_obstacle_detection(frame_data['rgb'])
                    
                    with obstacle_results_lock:
                        obstacle_results = basic_result
                
                processing_time = time.time() - start_time
                logger.debug(f"Depth analysis completed in {processing_time:.3f}s")
                
            except Exception as e:
                logger.error(f"Depth analysis failed: {e}")
                with obstacle_results_lock:
                    obstacle_results = {
                        'obstacle_detected': False,
                        'status': f"Depth analysis error: {e}"
                    }
        
        time.sleep(0.1)

def _basic_obstacle_detection(rgb_frame):
    """Temel engel algılama (ZED olmadığında)"""
    # Simple obstacle detection using computer vision
    # This is a placeholder - implement based on your needs
    return {
        'obstacle_detected': False,
        'distance': 'N/A',
        'status': 'Basic obstacle detection (no depth data)'
    }

def road_processor_thread():
    global detection_results, lane_results, obstacle_results, direction_data
    
    road_processor = RoadProcessor()
    
    while True:
        start_time = time.time()
        
        try:
            with detection_results_lock:
                det = detection_results.copy()
            with lane_results_lock:
                lanes = lane_results.copy()
            with obstacle_results_lock:
                obs = obstacle_results.copy()
            
            combined_data = {
                "detections": det,
                "lanes": lanes,
                "obstacles": obs,
                "timestamp": time.time()
            }
            
            dir_data = road_processor.process_road(combined_data)
            
            with direction_data_lock:
                direction_data = dir_data
            
            processing_time = time.time() - start_time
            logger.debug(f"Road processing completed in {processing_time:.3f}s")
            
        except Exception as e:
            logger.error(f"Road processing failed: {e}")
        
        time.sleep(0.1)

def direction_controller_thread():
    global direction_data
    
    direction_controller = DirectionController()
    
    while True:
        try:
            with direction_data_lock:
                dir_data = direction_data.copy() if direction_data else None
            
            if dir_data:
                control_result = direction_controller.control(dir_data)
                
                # Update direction data with control result
                with direction_data_lock:
                    direction_data.update(control_result)
                
        except Exception as e:
            logger.error(f"Direction control failed: {e}")
        
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
                    # Validate command through safety monitor
                    safe_command = safety_monitor.validate_control_command(dir_data)
                    arduino_communicator.send_data(safe_command)
            else:
                with arduino_status_lock:
                    arduino_status = "Disconnected"
                arduino_communicator.connect()
                
        except Exception as e:
            with arduino_status_lock:
                arduino_status = f"Error: {e}"
            logger.error(f"Arduino Communication Error: {e}")
        
        time.sleep(1)

def start_processing_threads():
    """Tüm işleme thread'lerini başlat"""
    logger.info("Starting enhanced processing system...")
    
    # Start performance monitoring
    memory_manager.start_monitoring()
    async_processor.start()
    
    # Register safety callbacks
    safety_monitor.register_health_callback(
        SystemComponent.CAMERA, 
        create_camera_health_callback()
    )
    safety_monitor.register_health_callback(
        SystemComponent.ARDUINO, 
        create_arduino_health_callback()
    )
    safety_monitor.register_health_callback(
        SystemComponent.PROCESSING, 
        create_processing_health_callback()
    )
    
    # Start safety monitoring
    safety_monitor.start_monitoring()
    
    # Start processing threads
    threads = [
        threading.Thread(target=camera_stream_thread, daemon=True, name="CameraStream"),
        threading.Thread(target=yolo_processor_thread, daemon=True, name="YOLOProcessor"),
        threading.Thread(target=lane_detector_thread, daemon=True, name="LaneDetector"),
        threading.Thread(target=depth_analyzer_thread, daemon=True, name="DepthAnalyzer"),
        threading.Thread(target=road_processor_thread, daemon=True, name="RoadProcessor"),
        threading.Thread(target=direction_controller_thread, daemon=True, name="DirectionController"),
        threading.Thread(target=arduino_communicator_thread, daemon=True, name="ArduinoComm")
    ]
    
    for thread in threads:
        thread.start()
        logger.info(f"Started thread: {thread.name}")
    
    logger.info("All processing threads started successfully")

def main():
    """Ana fonksiyon"""
    logger.info("=== Dursun Enhanced System Starting ===")
    
    try:
        start_processing_threads()
        logger.info("Enhanced system initialized. Press Ctrl+C to exit.")
        
        # Main monitoring loop
        while True:
            time.sleep(1)
            
            # Log system status periodically
            if int(time.time()) % 30 == 0:  # Every 30 seconds
                with performance_metrics_lock:
                    fps = performance_metrics['fps']
                
                safety_status = safety_monitor.get_safety_status()
                logger.info(f"System Status - FPS: {fps:.1f}, Safety: {safety_status['current_state']}")
                
    except KeyboardInterrupt:
        logger.info("Shutting down enhanced system...")
        
        # Cleanup
        safety_monitor.stop_monitoring()
        async_processor.stop()
        memory_manager.stop_monitoring()
        
        logger.info("Enhanced system shutdown complete")

if __name__ == "__main__":
    main()