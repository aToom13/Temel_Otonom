"""
Gelişmiş kamera yönetimi modülü.
ZED 2i kamera ve IMU entegrasyonu ile fallback mekanizması.
"""
import cv2
import numpy as np
import time
import threading
import logging
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass
from modules.imu_processor import IMUProcessor, VehicleMotion
import os
import yaml

logger = logging.getLogger(__name__)

@dataclass
class CameraFrame:
    """Kamera frame verisi"""
    rgb: np.ndarray
    depth: Optional[np.ndarray] = None
    confidence: Optional[np.ndarray] = None
    timestamp: float = 0.0
    camera_type: str = "unknown"
    resolution: Tuple[int, int] = (0, 0)

@dataclass
class CameraStatus:
    """Kamera durumu"""
    is_connected: bool
    camera_type: str  # "ZED", "Webcam", "None"
    resolution: Tuple[int, int]
    fps: float
    has_depth: bool
    has_imu: bool
    error_message: str = ""

class EnhancedCameraManager:
    """
    Gelişmiş kamera yönetimi sınıfı.
    - ZED 2i kamera öncelikli
    - Otomatik webcam fallback
    - IMU entegrasyonu
    - Hot-swap desteği
    """
    
    def __init__(self, config_path: str = None):
        self.config = self._load_config(config_path)
        
        # Camera instances
        self.zed_camera = None
        self.webcam = None
        self.current_camera_type = "None"
        
        # IMU processor
        self.imu_processor = IMUProcessor(config_path)
        
        # Frame data
        self.current_frame = None
        self.frame_lock = threading.Lock()
        
        # Status
        self.camera_status = CameraStatus(
            is_connected=False,
            camera_type="None",
            resolution=(0, 0),
            fps=0.0,
            has_depth=False,
            has_imu=False
        )
        
        # Monitoring
        self.monitoring_active = False
        self.monitor_thread = None
        
        # Performance metrics
        self.frame_count = 0
        self.last_fps_time = time.time()
        self.current_fps = 0.0
        
        logger.info("Enhanced Camera Manager initialized")
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Konfigürasyon dosyasını yükle"""
        if config_path is None:
            config_path = os.path.join(os.path.dirname(__file__), '..', 'config.yaml')
        
        default_config = {
            'zed_resolution': 'HD720',
            'zed_fps': 30,
            'fallback_webcam_index': 0,
            'auto_reconnect': True,
            'reconnect_interval': 5.0,
            'frame_timeout': 1.0,
            'depth_mode': 'PERFORMANCE',
            'coordinate_units': 'METER'
        }
        
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)
                    camera_config = config.get('camera', {})
                    default_config.update(camera_config)
            except Exception as e:
                logger.warning(f"Config loading failed: {e}, using defaults")
        
        return default_config
    
    def start(self) -> bool:
        """Kamera sistemini başlat"""
        try:
            # Try ZED camera first
            if self._init_zed_camera():
                self.current_camera_type = "ZED"
                logger.info("ZED camera initialized successfully")
            elif self._init_webcam():
                self.current_camera_type = "Webcam"
                logger.info("Webcam initialized as fallback")
            else:
                logger.error("No camera available")
                return False
            
            # Start monitoring thread
            self.monitoring_active = True
            self.monitor_thread = threading.Thread(
                target=self._monitoring_loop,
                daemon=True
            )
            self.monitor_thread.start()
            
            return True
            
        except Exception as e:
            logger.error(f"Camera system startup failed: {e}")
            return False
    
    def stop(self):
        """Kamera sistemini durdur"""
        self.monitoring_active = False
        
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2.0)
        
        self._cleanup_cameras()
        logger.info("Camera system stopped")
    
    def _init_zed_camera(self) -> bool:
        """ZED kamerayı başlat"""
        try:
            import pyzed.sl as sl
            
            self.zed_camera = sl.Camera()
            init_params = sl.InitParameters()
            
            # Resolution settings
            resolution_map = {
                'HD720': sl.RESOLUTION.HD720,
                'HD1080': sl.RESOLUTION.HD1080,
                'HD2K': sl.RESOLUTION.HD2K
            }
            
            resolution_str = self.config.get('zed_resolution', 'HD720')
            init_params.camera_resolution = resolution_map.get(resolution_str, sl.RESOLUTION.HD720)
            init_params.camera_fps = self.config.get('zed_fps', 30)
            
            # Depth settings
            depth_mode_map = {
                'PERFORMANCE': sl.DEPTH_MODE.PERFORMANCE,
                'QUALITY': sl.DEPTH_MODE.QUALITY,
                'ULTRA': sl.DEPTH_MODE.ULTRA
            }
            depth_mode = self.config.get('depth_mode', 'PERFORMANCE')
            init_params.depth_mode = depth_mode_map.get(depth_mode, sl.DEPTH_MODE.PERFORMANCE)
            
            # Coordinate system
            coordinate_units_map = {
                'METER': sl.UNIT.METER,
                'CENTIMETER': sl.UNIT.CENTIMETER,
                'MILLIMETER': sl.UNIT.MILLIMETER
            }
            units = self.config.get('coordinate_units', 'METER')
            init_params.coordinate_units = coordinate_units_map.get(units, sl.UNIT.METER)
            
            # Enable sensors
            init_params.sensors_required = True
            
            # Open camera
            err = self.zed_camera.open(init_params)
            if err != sl.ERROR_CODE.SUCCESS:
                logger.error(f"ZED Camera Error: {err}")
                self.zed_camera = None
                return False
            
            # Get camera info
            camera_info = self.zed_camera.get_camera_information()
            resolution = (
                camera_info.camera_configuration.resolution.width,
                camera_info.camera_configuration.resolution.height
            )
            
            # Update status
            self.camera_status = CameraStatus(
                is_connected=True,
                camera_type="ZED",
                resolution=resolution,
                fps=init_params.camera_fps,
                has_depth=True,
                has_imu=True
            )
            
            return True
            
        except ImportError:
            logger.warning("pyzed not found. ZED camera not available.")
            return False
        except Exception as e:
            logger.error(f"ZED camera initialization failed: {e}")
            return False
    
    def _init_webcam(self) -> bool:
        """Webcam'i başlat ve uygun bir tane bul"""
        try:
            # Birden fazla kamera indeksini ve backend'i dene
            for i in range(5):  # Genellikle ilk birkaç indekste bulunur
                try:
                    # CAP_DSHOW, Windows'ta daha kararlı çalışabilir
                    cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
                    if cap.isOpened():
                        # Test frame capture
                        ret, frame = cap.read()
                        if ret and frame is not None:
                            logger.info(f"Webcam found at index {i} using DSHOW backend")
                            self.webcam = cap
                            self.config['fallback_webcam_index'] = i  # Bulunan indeksi kaydet
                            break
                        else:
                            cap.release()
                    else:
                        cap.release()
                except Exception as e:
                    logger.debug(f"Failed to open camera at index {i}: {e}")
                    continue
            else:
                # Fallback to default backend
                try:
                    cap = cv2.VideoCapture(0)
                    if cap.isOpened():
                        ret, frame = cap.read()
                        if ret and frame is not None:
                            logger.info("Webcam found using default backend")
                            self.webcam = cap
                        else:
                            cap.release()
                            logger.error("No available webcam found")
                            return False
                    else:
                        logger.error("No available webcam found")
                        return False
                except Exception as e:
                    logger.error(f"Webcam initialization failed: {e}")
                    return False

            if not self.webcam or not self.webcam.isOpened():
                logger.error("Webcam could not be opened")
                return False
            
            # Set webcam properties
            self.webcam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.webcam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.webcam.set(cv2.CAP_PROP_FPS, 30)
            
            # Get actual resolution
            width = int(self.webcam.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.webcam.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = self.webcam.get(cv2.CAP_PROP_FPS)
            
            # Update status
            self.camera_status = CameraStatus(
                is_connected=True,
                camera_type="Webcam",
                resolution=(width, height),
                fps=fps,
                has_depth=False,
                has_imu=False
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Webcam initialization failed: {e}")
            return False
    
    def capture_frame(self) -> Optional[CameraFrame]:
        """Frame yakala"""
        try:
            if self.current_camera_type == "ZED":
                return self._capture_zed_frame()
            elif self.current_camera_type == "Webcam":
                return self._capture_webcam_frame()
            else:
                return None
                
        except Exception as e:
            logger.error(f"Frame capture failed: {e}")
            return None
    
    def _capture_zed_frame(self) -> Optional[CameraFrame]:
        """ZED frame yakala"""
        if not self.zed_camera:
            return None
        
        try:
            import pyzed.sl as sl
            
            runtime_parameters = sl.RuntimeParameters()
            
            if self.zed_camera.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
                # RGB image
                image = sl.Mat()
                self.zed_camera.retrieve_image(image, sl.VIEW.LEFT)
                rgb_frame = image.get_data()
                # Convert BGRA to BGR if needed
                if rgb_frame.shape[2] == 4:
                    rgb_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_BGRA2BGR)
                
                # Depth map
                depth = sl.Mat()
                self.zed_camera.retrieve_measure(depth, sl.MEASURE.DEPTH)
                depth_frame = depth.get_data()
                
                # Confidence map
                confidence = sl.Mat()
                self.zed_camera.retrieve_measure(confidence, sl.MEASURE.CONFIDENCE)
                confidence_frame = confidence.get_data()
                
                # IMU data
                imu_data = sl.SensorsData()
                if self.zed_camera.get_sensors_data(imu_data, sl.TIME_REFERENCE.IMAGE) == sl.ERROR_CODE.SUCCESS:
                    # Process IMU data
                    imu_raw = imu_data.get_imu_data()

                    def _safe_vec(v):
                        try:
                            return (float(v[0]), float(v[1]), float(v[2]))
                        except Exception:
                            return (0.0, 0.0, 0.0)

                    # Retrieve linear acceleration
                    if hasattr(imu_raw, "get_linear_acceleration"):
                        lin_accel = imu_raw.get_linear_acceleration()
                    else:
                        lin_accel = getattr(imu_raw, "linear_acceleration", None)

                    # Retrieve angular velocity
                    if hasattr(imu_raw, "get_angular_velocity"):
                        ang_vel = imu_raw.get_angular_velocity()
                    else:
                        ang_vel = getattr(imu_raw, "angular_velocity", None)

                    # Retrieve orientation (quaternion)
                    if hasattr(imu_raw, "get_orientation"):
                        orientation = imu_raw.get_orientation()
                    else:
                        orientation = getattr(imu_raw, "orientation", None)

                    if lin_accel is not None and ang_vel is not None and orientation is not None:
                        imu_dict = {
                            'linear_acceleration': {
                                'x': _safe_vec(lin_accel)[0],
                                'y': _safe_vec(lin_accel)[1],
                                'z': _safe_vec(lin_accel)[2]
                            },
                            'angular_velocity': {
                                'x': _safe_vec(ang_vel)[0],
                                'y': _safe_vec(ang_vel)[1],
                                'z': _safe_vec(ang_vel)[2]
                            },
                            'orientation': {
                                'x': orientation[0] if len(orientation) >= 4 else 0.0,
                                'y': orientation[1] if len(orientation) >= 4 else 0.0,
                                'z': orientation[2] if len(orientation) >= 4 else 0.0,
                                'w': orientation[3] if len(orientation) >= 4 else 1.0
                            }
                        }
                        # Process IMU data if we have everything we need
                        self.imu_processor.process_imu_data(imu_dict)
                
                # Update performance metrics
                self._update_fps()
                
                return CameraFrame(
                    rgb=rgb_frame,
                    depth=depth_frame,
                    confidence=confidence_frame,
                    timestamp=time.time(),
                    camera_type="ZED",
                    resolution=self.camera_status.resolution
                )
            
            return None
            
        except Exception as e:
            logger.error(f"ZED frame capture failed: {e}")
            return None
    
    def _capture_webcam_frame(self) -> Optional[CameraFrame]:
        """Webcam frame yakala"""
        if not self.webcam:
            return None
        
        try:
            ret, frame = self.webcam.read()
            if not ret:
                logger.warning("Failed to capture webcam frame")
                return None
            
            # Update performance metrics
            self._update_fps()
            
            return CameraFrame(
                rgb=frame,
                depth=None,
                confidence=None,
                timestamp=time.time(),
                camera_type="Webcam",
                resolution=self.camera_status.resolution
            )
            
        except Exception as e:
            logger.error(f"Webcam frame capture failed: {e}")
            return None
    
    def _update_fps(self):
        """FPS hesapla"""
        self.frame_count += 1
        current_time = time.time()
        
        if current_time - self.last_fps_time >= 1.0:
            self.current_fps = self.frame_count / (current_time - self.last_fps_time)
            self.camera_status.fps = self.current_fps
            self.frame_count = 0
            self.last_fps_time = current_time
    
    def _monitoring_loop(self):
        """Kamera izleme döngüsü"""
        while self.monitoring_active:
            try:
                # Check camera health
                if not self._check_camera_health():
                    logger.warning("Camera health check failed, attempting reconnection")
                    self._attempt_reconnection()
                
                time.sleep(self.config.get('reconnect_interval', 5.0))
                
            except Exception as e:
                logger.error(f"Camera monitoring error: {e}")
                time.sleep(1.0)
    
    def _check_camera_health(self) -> bool:
        """Kamera sağlığını kontrol et"""
        if self.current_camera_type == "ZED":
            return self.zed_camera is not None
        elif self.current_camera_type == "Webcam":
            return self.webcam is not None and self.webcam.isOpened()
        return False
    
    def _attempt_reconnection(self):
        """Yeniden bağlanma dene"""
        if not self.config.get('auto_reconnect', True):
            return
        
        logger.info("Attempting camera reconnection...")
        
        # Cleanup current camera
        self._cleanup_cameras()
        
        # Try ZED first, then webcam
        if self._init_zed_camera():
            self.current_camera_type = "ZED"
            logger.info("Reconnected to ZED camera")
        elif self._init_webcam():
            self.current_camera_type = "Webcam"
            logger.info("Reconnected to webcam")
        else:
            self.current_camera_type = "None"
            self.camera_status.is_connected = False
            logger.error("Reconnection failed")
    
    def _cleanup_cameras(self):
        """Kameraları temizle"""
        if self.zed_camera:
            try:
                self.zed_camera.close()
            except:
                pass
            self.zed_camera = None
        
        if self.webcam:
            try:
                self.webcam.release()
            except:
                pass
            self.webcam = None
    
    def get_camera_status(self) -> CameraStatus:
        """Kamera durumunu al"""
        return self.camera_status
    
    def get_imu_data(self) -> Dict[str, Any]:
        """IMU verilerini al"""
        return self.imu_processor.get_motion_summary()
    
    def has_depth_capability(self) -> bool:
        """Derinlik özelliği var mı?"""
        return self.camera_status.has_depth
    
    def has_imu_capability(self) -> bool:
        """IMU özelliği var mı?"""
        return self.camera_status.has_imu
    
    def switch_to_zed_if_available(self) -> bool:
        """ZED kamera varsa ona geç"""
        if self.current_camera_type == "ZED":
            return True
        
        if self._init_zed_camera():
            # Cleanup webcam
            if self.webcam:
                self.webcam.release()
                self.webcam = None
            
            self.current_camera_type = "ZED"
            logger.info("Switched to ZED camera")
            return True
        
        return False