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
from modules.enhanced_camera_manager import EnhancedCameraManager
from modules.lidar_processor import RPLidarA1Processor

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
    log_file = log_conf.get('file', 'logs/dursun.log')
else:
    log_level = logging.INFO
    log_file = 'logs/dursun.log'

# Create logs directory if it doesn't exist
os.makedirs(os.path.dirname(log_file), exist_ok=True)

# Enhanced logging configuration
logging.basicConfig(
    level=log_level,
    format='%(asctime)s [%(levelname)s] %(threadName)s: %(message)s',
    handlers=[
        logging.FileHandler(log_file, encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Global variables for sharing data between threads
camera_manager = None
lidar_processor = None
processed_frame = None
processed_frame_lock = threading.Lock()
detection_results = {}
detection_results_lock = threading.Lock()
lane_results = {}
lane_results_lock = threading.Lock()
obstacle_results = {}
obstacle_results_lock = threading.Lock()
lidar_results = {}
lidar_results_lock = threading.Lock()
direction_data = {}
direction_data_lock = threading.Lock()
arduino_status = "Disconnected"
arduino_status_lock = threading.Lock()
imu_data = {}
imu_data_lock = threading.Lock()

# Performance metrics
performance_metrics = {
    'frame_count': 0,
    'processing_times': [],
    'fps': 0.0,
    'last_fps_update': time.time()
}
performance_metrics_lock = threading.Lock()

# System initialization flag
system_initialized = False
initialization_lock = threading.Lock()

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
            if len(performance_metrics['processing_times']) > 0:
                performance_metrics['fps'] = len(performance_metrics['processing_times']) / (
                    current_time - performance_metrics['last_fps_update']
                )
            performance_metrics['last_fps_update'] = current_time

def create_camera_health_callback():
    """Kamera sağlık kontrolü callback'i oluştur"""
    def check_camera_health():
        try:
            if camera_manager:
                camera_status = camera_manager.get_camera_status()
                
                if camera_status.is_connected:
                    return HealthStatus(
                        component=SystemComponent.CAMERA,
                        status=SafetyState.SAFE,
                        message=f"{camera_status.camera_type} camera operating normally",
                        timestamp=time.time(),
                        metrics={
                            "camera_type": camera_status.camera_type,
                            "resolution": camera_status.resolution,
                            "fps": camera_status.fps,
                            "has_depth": camera_status.has_depth,
                            "has_imu": camera_status.has_imu
                        }
                    )
                else:
                    return HealthStatus(
                        component=SystemComponent.CAMERA,
                        status=SafetyState.CRITICAL,
                        message="Camera disconnected",
                        timestamp=time.time(),
                        metrics={"camera_type": "None"}
                    )
            
            return HealthStatus(
                component=SystemComponent.CAMERA,
                status=SafetyState.CRITICAL,
                message="Camera manager not initialized",
                timestamp=time.time(),
                metrics={}
            )
        except Exception as e:
            logger.error(f"Camera health check failed: {e}")
            return HealthStatus(
                component=SystemComponent.CAMERA,
                status=SafetyState.CRITICAL,
                message=f"Camera health check error: {e}",
                timestamp=time.time(),
                metrics={}
            )
    
    return check_camera_health

def create_arduino_health_callback():
    """Arduino sağlık kontrolü callback'i oluştur"""
    def check_arduino_health():
        try:
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
                    status=SafetyState.WARNING,
                    message=f"Arduino issue: {status}",
                    timestamp=time.time(),
                    metrics={"status": status}
                )
        except Exception as e:
            logger.error(f"Arduino health check failed: {e}")
            return HealthStatus(
                component=SystemComponent.ARDUINO,
                status=SafetyState.CRITICAL,
                message=f"Arduino health check error: {e}",
                timestamp=time.time(),
                metrics={}
            )
    
    return check_arduino_health

def create_lidar_health_callback():
    """LiDAR sağlık kontrolü callback'i oluştur"""
    def check_lidar_health():
        try:
            if lidar_processor:
                lidar_status = lidar_processor.get_status()
                
                if lidar_status['is_connected'] and lidar_status['is_scanning']:
                    return HealthStatus(
                        component=SystemComponent.PROCESSING,
                        status=SafetyState.SAFE,
                        message="LiDAR operating normally",
                        timestamp=time.time(),
                        metrics={
                            "scan_frequency": lidar_status['scan_frequency'],
                            "obstacle_count": lidar_status['obstacle_count'],
                            "scan_count": lidar_status['scan_count']
                        }
                    )
                elif lidar_status['is_connected']:
                    return HealthStatus(
                        component=SystemComponent.PROCESSING,
                        status=SafetyState.WARNING,
                        message="LiDAR connected but not scanning",
                        timestamp=time.time(),
                        metrics=lidar_status
                    )
                else:
                    return HealthStatus(
                        component=SystemComponent.PROCESSING,
                        status=SafetyState.WARNING,
                        message="LiDAR not connected",
                        timestamp=time.time(),
                        metrics=lidar_status
                    )
            
            return HealthStatus(
                component=SystemComponent.PROCESSING,
                status=SafetyState.WARNING,
                message="LiDAR processor not initialized",
                timestamp=time.time(),
                metrics={}
            )
        except Exception as e:
            logger.error(f"LiDAR health check failed: {e}")
            return HealthStatus(
                component=SystemComponent.PROCESSING,
                status=SafetyState.WARNING,
                message=f"LiDAR health check error: {e}",
                timestamp=time.time(),
                metrics={}
            )
    
    return check_lidar_health

def create_processing_health_callback():
    """İşleme sağlık kontrolü callback'i oluştur"""
    def check_processing_health():
        try:
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
        except Exception as e:
            logger.error(f"Processing health check failed: {e}")
            return HealthStatus(
                component=SystemComponent.PROCESSING,
                status=SafetyState.CRITICAL,
                message=f"Processing health check error: {e}",
                timestamp=time.time(),
                metrics={}
            )
    
    return check_processing_health

# --- Enhanced Camera Stream ---
def camera_stream_thread():
    global camera_manager, processed_frame
    
    try:
        camera_manager = EnhancedCameraManager()
        
        if not camera_manager.start():
            logger.error("Failed to start camera system")
            return
        
        logger.info("Enhanced camera system started")
        
        while True:
            try:
                frame_data = camera_manager.capture_frame()
                
                if frame_data:
                    with processed_frame_lock:
                        processed_frame = frame_data.rgb
                    
                    # Update IMU data if available
                    if camera_manager.has_imu_capability():
                        current_imu_data = camera_manager.get_imu_data()
                        with imu_data_lock:
                            imu_data.update(current_imu_data)
                
                time.sleep(0.01)  # ~100 FPS max
                
            except Exception as e:
                logger.error(f"Camera stream error: {e}")
                time.sleep(0.1)
                
    except Exception as e:
        logger.error(f"Camera stream thread failed: {e}")

# --- LiDAR Processing Thread ---
def lidar_processor_thread():
    global lidar_processor, lidar_results
    
    try:
        lidar_processor = RPLidarA1Processor()
        
        # Try to connect to LiDAR
        if lidar_processor.connect():
            logger.info("LiDAR connected successfully")
            
            # Start scanning
            if lidar_processor.start_scanning():
                logger.info("LiDAR scanning started")
            else:
                logger.error("Failed to start LiDAR scanning")
                return
        else:
            logger.warning("LiDAR not connected, continuing without LiDAR")
            return
        
        while True:
            try:
                # Get LiDAR data for visualization and processing
                lidar_data = lidar_processor.get_scan_data_for_visualization()
                
                with lidar_results_lock:
                    lidar_results = lidar_data
                
                time.sleep(0.1)  # 10 Hz
                
            except Exception as e:
                logger.error(f"LiDAR processing error: {e}")
                time.sleep(0.5)
                
    except Exception as e:
        logger.error(f"LiDAR processor thread failed: {e}")

# --- Enhanced Processing Threads ---
def yolo_processor_thread():
    global processed_frame, detection_results
    
    try:
        yolo_processor = YoloProcessor()
        
        while True:
            start_time = time.time()
            
            with processed_frame_lock:
                frame = processed_frame.copy() if processed_frame is not None else None
            
            if frame is not None:
                try:
                    proc_frame, det_results = yolo_processor.process_frame(frame)
                    
                    with processed_frame_lock:
                        processed_frame = proc_frame
                    with detection_results_lock:
                        detection_results = det_results
                    
                    processing_time = time.time() - start_time
                    update_performance_metrics(processing_time)
                    
                except Exception as e:
                    logger.error(f"YOLO processing failed: {e}")
            
            time.sleep(0.1)
            
    except Exception as e:
        logger.error(f"YOLO processor thread failed: {e}")

def lane_detector_thread():
    global processed_frame, lane_results
    
    try:
        lane_detector = EnhancedLaneDetector()
        
        while True:
            start_time = time.time()
            
            with processed_frame_lock:
                frame = processed_frame.copy() if processed_frame is not None else None
            
            if frame is not None:
                try:
                    lane_res = lane_detector.detect_lanes_with_tracking(frame)
                    
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
            
    except Exception as e:
        logger.error(f"Lane detector thread failed: {e}")

def depth_analyzer_thread():
    global camera_manager, obstacle_results
    
    try:
        # Import advanced depth processor
        from core.algorithms.advanced_depth_processing import AdvancedDepthProcessor
        depth_analyzer = AdvancedDepthProcessor()
        
        while True:
            start_time = time.time()
            
            if camera_manager:
                try:
                    frame_data = camera_manager.capture_frame()
                    
                    if frame_data and camera_manager.has_depth_capability() and frame_data.depth is not None:
                        # Use advanced depth processing
                        depth_result = depth_analyzer.process_depth_map(
                            frame_data.depth, 
                            frame_data.confidence
                        )
                        
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
                    elif frame_data and frame_data.rgb is not None:
                        # Fallback to basic analysis
                        basic_result = _basic_obstacle_detection(frame_data.rgb)
                        
                        with obstacle_results_lock:
                            obstacle_results = basic_result
                    else:
                        with obstacle_results_lock:
                            obstacle_results = {
                                'obstacle_detected': False,
                                'obstacle_count': 0,
                                'status': 'No camera data available'
                            }
                    
                    processing_time = time.time() - start_time
                    logger.debug(f"Depth analysis completed in {processing_time:.3f}s")
                    
                except Exception as e:
                    logger.error(f"Depth analysis failed: {e}")
                    with obstacle_results_lock:
                        obstacle_results = {
                            'obstacle_detected': False,
                            'obstacle_count': 0,
                            'status': f"Depth analysis error: {e}"
                        }
            
            time.sleep(0.1)
            
    except Exception as e:
        logger.error(f"Depth analyzer thread failed: {e}")

def _basic_obstacle_detection(rgb_frame):
    """Temel engel algılama (ZED olmadığında)"""
    return {
        'obstacle_detected': False,
        'obstacle_count': 0,
        'distance': 'N/A',
        'status': 'Basic obstacle detection (no depth data)'
    }

def road_processor_thread():
    global detection_results, lane_results, obstacle_results, lidar_results, direction_data, imu_data
    
    try:
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
                with lidar_results_lock:
                    lidar = lidar_results.copy()
                with imu_data_lock:
                    imu = imu_data.copy()
                
                combined_data = {
                    "detections": det,
                    "lanes": lanes,
                    "obstacles": obs,
                    "lidar": lidar,
                    "imu": imu,
                    "timestamp": time.time()
                }
                
                dir_data = road_processor.process_road(combined_data)
                
                # Add IMU-based enhancements
                if imu and camera_manager and camera_manager.has_imu_capability():
                    dir_data['vehicle_heading'] = imu.get('heading_degrees', 0.0)
                    dir_data['vehicle_tilt'] = {
                        'roll': imu.get('roll_degrees', 0.0),
                        'pitch': imu.get('pitch_degrees', 0.0)
                    }
                    dir_data['is_moving'] = imu.get('is_moving', False)
                    dir_data['speed_estimate'] = imu.get('speed_kmh', 0.0)
                    dir_data['motion_confidence'] = imu.get('motion_confidence', 0.0)
                
                # Add LiDAR-based enhancements
                if lidar and lidar.get('is_scanning', False):
                    dir_data['lidar_obstacle_count'] = len(lidar.get('obstacles', []))
                    dir_data['lidar_scan_quality'] = lidar.get('scan_frequency', 0.0)
                
                with direction_data_lock:
                    direction_data = dir_data
                
                processing_time = time.time() - start_time
                logger.debug(f"Road processing completed in {processing_time:.3f}s")
                
            except Exception as e:
                logger.error(f"Road processing failed: {e}")
            
            time.sleep(0.1)
            
    except Exception as e:
        logger.error(f"Road processor thread failed: {e}")

def direction_controller_thread():
    global direction_data
    
    try:
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
            
    except Exception as e:
        logger.error(f"Direction controller thread failed: {e}")

def arduino_communicator_thread():
    global direction_data, arduino_status
    
    try:
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
            
    except Exception as e:
        logger.error(f"Arduino communicator thread failed: {e}")

def start_processing_threads():
    """Tüm işleme thread'lerini başlat"""
    global system_initialized
    
    with initialization_lock:
        if system_initialized:
            logger.warning("System already initialized")
            return
        
        logger.info("Starting enhanced processing system...")
        
        try:
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
                threading.Thread(target=lidar_processor_thread, daemon=True, name="LiDARProcessor"),
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
            
            system_initialized = True
            logger.info("All processing threads started successfully")
            
        except Exception as e:
            logger.error(f"Failed to start processing threads: {e}")
            raise

def get_system_status():
    """Sistem durumunu al"""
    try:
        camera_status = camera_manager.get_camera_status() if camera_manager else None
        lidar_status = lidar_processor.get_status() if lidar_processor else None
        
        with arduino_status_lock:
            arduino_stat = arduino_status
        with detection_results_lock:
            det_results = detection_results.copy()
        with lane_results_lock:
            lane_res = lane_results.copy()
        with obstacle_results_lock:
            obs_results = obstacle_results.copy()
        with lidar_results_lock:
            lidar_res = lidar_results.copy()
        with direction_data_lock:
            dir_data = direction_data.copy()
        with imu_data_lock:
            imu_dat = imu_data.copy()
        with performance_metrics_lock:
            perf_metrics = performance_metrics.copy()
        
        return {
            "camera_status": {
                "is_connected": camera_status.is_connected if camera_status else False,
                "camera_type": camera_status.camera_type if camera_status else "None",
                "resolution": camera_status.resolution if camera_status else (0, 0),
                "fps": camera_status.fps if camera_status else 0.0,
                "has_depth": camera_status.has_depth if camera_status else False,
                "has_imu": camera_status.has_imu if camera_status else False
            },
            "lidar_status": lidar_status or {
                "is_connected": False,
                "is_scanning": False,
                "scan_frequency": 0.0,
                "obstacle_count": 0
            },
            "arduino_status": arduino_stat,
            "detection_results": det_results,
            "lane_results": lane_res,
            "obstacle_results": obs_results,
            "lidar_results": lidar_res,
            "direction_data": dir_data,
            "imu_data": imu_dat,
            "safety_status": safety_monitor.get_safety_status(),
            "performance_metrics": perf_metrics,
            "timestamp": time.time()
        }
    except Exception as e:
        logger.error(f"Failed to get system status: {e}")
        return {
            "error": str(e),
            "timestamp": time.time()
        }

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
                try:
                    with performance_metrics_lock:
                        fps = performance_metrics['fps']
                    
                    safety_status = safety_monitor.get_safety_status()
                    camera_status = camera_manager.get_camera_status() if camera_manager else None
                    camera_type = camera_status.camera_type if camera_status else "None"
                    lidar_status = lidar_processor.get_status() if lidar_processor else {"is_connected": False}
                    lidar_connected = "Connected" if lidar_status.get('is_connected', False) else "Disconnected"
                    
                    logger.info(f"System Status - FPS: {fps:.1f}, Camera: {camera_type}, LiDAR: {lidar_connected}, Safety: {safety_status['current_state']}")
                except Exception as e:
                    logger.error(f"Status logging failed: {e}")
                
    except KeyboardInterrupt:
        logger.info("Shutting down enhanced system...")
        
        # Cleanup
        try:
            safety_monitor.stop_monitoring()
            async_processor.stop()
            memory_manager.stop_monitoring()
            
            if camera_manager:
                camera_manager.stop()
            
            if lidar_processor:
                lidar_processor.disconnect()
                
        except Exception as e:
            logger.error(f"Cleanup failed: {e}")
        
        logger.info("Enhanced system shutdown complete")
    except Exception as e:
        logger.error(f"System error: {e}")
        raise

if __name__ == "__main__":
    main()