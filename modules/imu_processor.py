"""
ZED 2i IMU sensör işleme modülü.
ZED kameranın dahili IMU sensöründen veri okuma ve işleme.
"""
import numpy as np
import time
import logging
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass
from collections import deque
import os
import yaml

logger = logging.getLogger(__name__)

@dataclass
class IMUData:
    """IMU sensör verisi"""
    timestamp: float
    acceleration: np.ndarray  # [x, y, z] m/s²
    angular_velocity: np.ndarray  # [x, y, z] rad/s
    orientation: np.ndarray  # [roll, pitch, yaw] radians
    linear_acceleration: np.ndarray  # Gravity compensated
    magnetic_field: Optional[np.ndarray] = None  # [x, y, z] µT

@dataclass
class VehicleMotion:
    """Araç hareket durumu"""
    velocity: np.ndarray  # [x, y, z] m/s
    position: np.ndarray  # [x, y, z] m
    orientation: np.ndarray  # [roll, pitch, yaw] rad
    angular_velocity: np.ndarray  # [roll_rate, pitch_rate, yaw_rate] rad/s
    acceleration: np.ndarray  # [x, y, z] m/s²
    is_moving: bool
    motion_confidence: float

class IMUProcessor:
    """
    ZED 2i IMU sensör işleme sınıfı.
    - IMU verisi okuma ve filtreleme
    - Sensor fusion (IMU + Camera)
    - Motion estimation
    - Orientation tracking
    """
    
    def __init__(self, config_path: str = None):
        self.config = self._load_config(config_path)
        
        # IMU data history
        self.imu_history = deque(maxsize=self.config.get('history_size', 100))
        self.motion_history = deque(maxsize=50)
        
        # Kalman filter for sensor fusion
        self.kalman_filter = self._init_kalman_filter()
        
        # Motion detection
        self.motion_threshold = self.config.get('motion_threshold', 0.1)  # m/s
        self.stationary_threshold = self.config.get('stationary_threshold', 0.05)  # m/s
        
        # Calibration
        self.gravity_vector = np.array([0, 0, -9.81])  # m/s²
        self.is_calibrated = False
        self.calibration_samples = []
        
        # Current state
        self.current_motion = VehicleMotion(
            velocity=np.zeros(3),
            position=np.zeros(3),
            orientation=np.zeros(3),
            angular_velocity=np.zeros(3),
            acceleration=np.zeros(3),
            is_moving=False,
            motion_confidence=0.0
        )
        
        logger.info("IMU Processor initialized")
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Konfigürasyon dosyasını yükle"""
        if config_path is None:
            config_path = os.path.join(os.path.dirname(__file__), '..', 'config.yaml')
        
        default_config = {
            'history_size': 100,
            'motion_threshold': 0.1,
            'stationary_threshold': 0.05,
            'kalman_process_noise': 0.01,
            'kalman_measurement_noise': 0.1,
            'calibration_samples': 100,
            'gravity_compensation': True,
            'orientation_filter_alpha': 0.98  # Complementary filter
        }
        
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)
                    imu_config = config.get('imu', {})
                    default_config.update(imu_config)
            except Exception as e:
                logger.warning(f"Config loading failed: {e}, using defaults")
        
        return default_config
    
    def _init_kalman_filter(self):
        """Kalman filtresi başlat"""
        # Simplified Kalman filter for orientation estimation
        # State: [roll, pitch, yaw, roll_rate, pitch_rate, yaw_rate]
        from scipy.linalg import block_diag
        
        # State transition matrix (6x6)
        dt = 0.01  # 100Hz assumption
        F = np.eye(6)
        F[0:3, 3:6] = np.eye(3) * dt
        
        # Process noise covariance
        process_noise = self.config.get('kalman_process_noise', 0.01)
        Q = np.eye(6) * process_noise
        
        # Measurement noise covariance
        measurement_noise = self.config.get('kalman_measurement_noise', 0.1)
        R = np.eye(3) * measurement_noise
        
        # Measurement matrix (3x6) - we measure orientation directly
        H = np.zeros((3, 6))
        H[0:3, 0:3] = np.eye(3)
        
        return {
            'F': F,
            'Q': Q,
            'R': R,
            'H': H,
            'x': np.zeros(6),  # State vector
            'P': np.eye(6) * 0.1  # Error covariance
        }
    
    def process_imu_data(self, zed_imu_data: Dict[str, Any]) -> VehicleMotion:
        """
        ZED IMU verisini işle ve araç hareket durumunu hesapla.
        
        Args:
            zed_imu_data: ZED kameradan gelen IMU verisi
            
        Returns:
            İşlenmiş araç hareket durumu
        """
        try:
            # ZED IMU verisini parse et
            imu_data = self._parse_zed_imu_data(zed_imu_data)
            
            # Kalibrasyon kontrolü
            if not self.is_calibrated:
                self._calibrate_imu(imu_data)
                return self.current_motion
            
            # IMU verisini filtrele
            filtered_imu = self._filter_imu_data(imu_data)
            
            # Gravity compensation
            if self.config.get('gravity_compensation', True):
                filtered_imu = self._compensate_gravity(filtered_imu)
            
            # Sensor fusion (IMU + previous estimates)
            fused_data = self._sensor_fusion(filtered_imu)
            
            # Motion estimation
            motion = self._estimate_motion(fused_data)
            
            # Motion detection
            motion = self._detect_motion_state(motion)
            
            # Update history
            self.imu_history.append(filtered_imu)
            self.motion_history.append(motion)
            
            self.current_motion = motion
            
            return motion
            
        except Exception as e:
            logger.error(f"IMU processing failed: {e}")
            return self.current_motion
    
    def _parse_zed_imu_data(self, zed_data: Dict[str, Any]) -> IMUData:
        """ZED IMU verisini parse et"""
        timestamp = time.time()
        
        # ZED IMU data structure'ına göre parse et
        if 'linear_acceleration' in zed_data:
            acceleration = np.array([
                zed_data['linear_acceleration']['x'],
                zed_data['linear_acceleration']['y'],
                zed_data['linear_acceleration']['z']
            ])
        else:
            acceleration = np.zeros(3)
        
        if 'angular_velocity' in zed_data:
            angular_velocity = np.array([
                zed_data['angular_velocity']['x'],
                zed_data['angular_velocity']['y'],
                zed_data['angular_velocity']['z']
            ])
        else:
            angular_velocity = np.zeros(3)
        
        if 'orientation' in zed_data:
            # Quaternion to Euler conversion
            orientation = self._quaternion_to_euler(
                zed_data['orientation']['x'],
                zed_data['orientation']['y'],
                zed_data['orientation']['z'],
                zed_data['orientation']['w']
            )
        else:
            orientation = np.zeros(3)
        
        return IMUData(
            timestamp=timestamp,
            acceleration=acceleration,
            angular_velocity=angular_velocity,
            orientation=orientation,
            linear_acceleration=acceleration  # Will be gravity compensated later
        )
    
    def _quaternion_to_euler(self, x: float, y: float, z: float, w: float) -> np.ndarray:
        """Quaternion'ı Euler açılarına çevir"""
        # Roll (x-axis rotation)
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = np.arctan2(sinr_cosp, cosr_cosp)
        
        # Pitch (y-axis rotation)
        sinp = 2 * (w * y - z * x)
        if abs(sinp) >= 1:
            pitch = np.copysign(np.pi / 2, sinp)  # Use 90 degrees if out of range
        else:
            pitch = np.arcsin(sinp)
        
        # Yaw (z-axis rotation)
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)
        
        return np.array([roll, pitch, yaw])
    
    def _calibrate_imu(self, imu_data: IMUData):
        """IMU kalibrasyonu"""
        self.calibration_samples.append(imu_data.acceleration)
        
        if len(self.calibration_samples) >= self.config.get('calibration_samples', 100):
            # Gravity vector'ü hesapla (stationary durumda)
            avg_acceleration = np.mean(self.calibration_samples, axis=0)
            self.gravity_vector = avg_acceleration
            self.is_calibrated = True
            
            logger.info(f"IMU calibrated. Gravity vector: {self.gravity_vector}")
    
    def _filter_imu_data(self, imu_data: IMUData) -> IMUData:
        """IMU verisini filtrele (noise reduction)"""
        if len(self.imu_history) == 0:
            return imu_data
        
        # Simple moving average filter
        window_size = min(5, len(self.imu_history))
        recent_data = list(self.imu_history)[-window_size:]
        
        # Average acceleration
        avg_acceleration = np.mean([data.acceleration for data in recent_data], axis=0)
        filtered_acceleration = 0.7 * imu_data.acceleration + 0.3 * avg_acceleration
        
        # Average angular velocity
        avg_angular_velocity = np.mean([data.angular_velocity for data in recent_data], axis=0)
        filtered_angular_velocity = 0.8 * imu_data.angular_velocity + 0.2 * avg_angular_velocity
        
        return IMUData(
            timestamp=imu_data.timestamp,
            acceleration=filtered_acceleration,
            angular_velocity=filtered_angular_velocity,
            orientation=imu_data.orientation,
            linear_acceleration=filtered_acceleration
        )
    
    def _compensate_gravity(self, imu_data: IMUData) -> IMUData:
        """Gravity compensation"""
        # Remove gravity from acceleration
        linear_acceleration = imu_data.acceleration - self.gravity_vector
        
        return IMUData(
            timestamp=imu_data.timestamp,
            acceleration=imu_data.acceleration,
            angular_velocity=imu_data.angular_velocity,
            orientation=imu_data.orientation,
            linear_acceleration=linear_acceleration
        )
    
    def _sensor_fusion(self, imu_data: IMUData) -> IMUData:
        """Kalman filter ile sensor fusion"""
        kf = self.kalman_filter
        dt = 0.01  # Assume 100Hz
        
        # Prediction step
        kf['F'][0:3, 3:6] = np.eye(3) * dt
        kf['x'] = kf['F'] @ kf['x']
        kf['P'] = kf['F'] @ kf['P'] @ kf['F'].T + kf['Q']
        
        # Update step with orientation measurement
        y = imu_data.orientation - kf['H'] @ kf['x']  # Innovation
        S = kf['H'] @ kf['P'] @ kf['H'].T + kf['R']  # Innovation covariance
        K = kf['P'] @ kf['H'].T @ np.linalg.inv(S)  # Kalman gain
        
        kf['x'] = kf['x'] + K @ y
        kf['P'] = (np.eye(6) - K @ kf['H']) @ kf['P']
        
        # Extract fused orientation and angular velocity
        fused_orientation = kf['x'][0:3]
        fused_angular_velocity = kf['x'][3:6]
        
        return IMUData(
            timestamp=imu_data.timestamp,
            acceleration=imu_data.acceleration,
            angular_velocity=fused_angular_velocity,
            orientation=fused_orientation,
            linear_acceleration=imu_data.linear_acceleration
        )
    
    def _estimate_motion(self, imu_data: IMUData) -> VehicleMotion:
        """Hareket durumunu tahmin et"""
        if len(self.motion_history) == 0:
            # İlk ölçüm
            return VehicleMotion(
                velocity=np.zeros(3),
                position=np.zeros(3),
                orientation=imu_data.orientation,
                angular_velocity=imu_data.angular_velocity,
                acceleration=imu_data.linear_acceleration,
                is_moving=False,
                motion_confidence=0.0
            )
        
        # Previous motion state
        prev_motion = self.motion_history[-1]
        dt = imu_data.timestamp - self.imu_history[-1].timestamp if self.imu_history else 0.01
        
        # Integrate acceleration to get velocity
        velocity = prev_motion.velocity + imu_data.linear_acceleration * dt
        
        # Integrate velocity to get position
        position = prev_motion.position + velocity * dt
        
        # Apply velocity damping (friction/air resistance)
        velocity_magnitude = np.linalg.norm(velocity)
        if velocity_magnitude > 0:
            damping_factor = 0.95  # Adjust based on vehicle characteristics
            velocity *= damping_factor
        
        return VehicleMotion(
            velocity=velocity,
            position=position,
            orientation=imu_data.orientation,
            angular_velocity=imu_data.angular_velocity,
            acceleration=imu_data.linear_acceleration,
            is_moving=False,  # Will be determined in next step
            motion_confidence=0.0
        )
    
    def _detect_motion_state(self, motion: VehicleMotion) -> VehicleMotion:
        """Hareket durumunu tespit et"""
        velocity_magnitude = np.linalg.norm(motion.velocity)
        acceleration_magnitude = np.linalg.norm(motion.acceleration)
        angular_velocity_magnitude = np.linalg.norm(motion.angular_velocity)
        
        # Motion detection
        is_moving = (
            velocity_magnitude > self.motion_threshold or
            acceleration_magnitude > 0.5 or  # m/s²
            angular_velocity_magnitude > 0.1  # rad/s
        )
        
        # Motion confidence based on multiple factors
        velocity_confidence = min(1.0, velocity_magnitude / 2.0)  # Normalize to 2 m/s
        acceleration_confidence = min(1.0, acceleration_magnitude / 2.0)
        angular_confidence = min(1.0, angular_velocity_magnitude / 1.0)
        
        motion_confidence = (velocity_confidence + acceleration_confidence + angular_confidence) / 3.0
        
        motion.is_moving = is_moving
        motion.motion_confidence = motion_confidence
        
        return motion
    
    def get_vehicle_heading(self) -> float:
        """Araç yönünü al (yaw açısı)"""
        return self.current_motion.orientation[2]  # Yaw angle
    
    def get_vehicle_tilt(self) -> Tuple[float, float]:
        """Araç eğimini al (roll, pitch)"""
        return self.current_motion.orientation[0], self.current_motion.orientation[1]
    
    def is_vehicle_stationary(self) -> bool:
        """Araç durgun mu?"""
        velocity_magnitude = np.linalg.norm(self.current_motion.velocity)
        return velocity_magnitude < self.stationary_threshold
    
    def get_motion_summary(self) -> Dict[str, Any]:
        """Hareket durumu özeti"""
        motion = self.current_motion
        
        return {
            'is_moving': motion.is_moving,
            'velocity_magnitude': float(np.linalg.norm(motion.velocity)),
            'speed_kmh': float(np.linalg.norm(motion.velocity) * 3.6),  # Convert m/s to km/h
            'heading_degrees': float(np.degrees(motion.orientation[2])),
            'roll_degrees': float(np.degrees(motion.orientation[0])),
            'pitch_degrees': float(np.degrees(motion.orientation[1])),
            'acceleration_magnitude': float(np.linalg.norm(motion.acceleration)),
            'angular_velocity_magnitude': float(np.linalg.norm(motion.angular_velocity)),
            'motion_confidence': float(motion.confidence),
            'is_calibrated': self.is_calibrated,
            'position': motion.position.tolist(),
            'velocity': motion.velocity.tolist(),
            'orientation_rad': motion.orientation.tolist(),
            'angular_velocity_rad': motion.angular_velocity.tolist()
        }