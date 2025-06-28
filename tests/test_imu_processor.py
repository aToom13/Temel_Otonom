import pytest
import numpy as np
import time
from unittest.mock import Mock, patch
from modules.imu_processor import IMUProcessor, IMUData, VehicleMotion

@pytest.fixture
def imu_processor():
    """Create IMU processor for testing"""
    return IMUProcessor()

@pytest.fixture
def sample_zed_imu_data():
    """Sample ZED IMU data"""
    return {
        'linear_acceleration': {'x': 0.1, 'y': 0.2, 'z': 9.8},
        'angular_velocity': {'x': 0.01, 'y': 0.02, 'z': 0.03},
        'orientation': {'x': 0.0, 'y': 0.0, 'z': 0.0, 'w': 1.0}
    }

def test_imu_processor_initialization(imu_processor):
    """Test IMU processor initialization"""
    assert imu_processor is not None
    assert not imu_processor.is_calibrated
    assert len(imu_processor.imu_history) == 0
    assert imu_processor.current_motion is not None

def test_config_loading(imu_processor):
    """Test configuration loading"""
    config = imu_processor.config
    
    assert 'history_size' in config
    assert 'motion_threshold' in config
    assert 'kalman_process_noise' in config
    assert 'gravity_compensation' in config

def test_zed_imu_data_parsing(imu_processor, sample_zed_imu_data):
    """Test ZED IMU data parsing"""
    imu_data = imu_processor._parse_zed_imu_data(sample_zed_imu_data)
    
    assert isinstance(imu_data, IMUData)
    assert imu_data.timestamp > 0
    assert len(imu_data.acceleration) == 3
    assert len(imu_data.angular_velocity) == 3
    assert len(imu_data.orientation) == 3

def test_quaternion_to_euler_conversion(imu_processor):
    """Test quaternion to Euler angle conversion"""
    # Test identity quaternion (no rotation)
    euler = imu_processor._quaternion_to_euler(0, 0, 0, 1)
    assert np.allclose(euler, [0, 0, 0], atol=1e-6)
    
    # Test 90-degree rotation around Z-axis
    euler = imu_processor._quaternion_to_euler(0, 0, 0.707, 0.707)
    assert abs(euler[2] - np.pi/2) < 0.01  # Yaw should be ~90 degrees

def test_imu_calibration(imu_processor, sample_zed_imu_data):
    """Test IMU calibration process"""
    # Add calibration samples
    for _ in range(imu_processor.config['calibration_samples']):
        imu_data = imu_processor._parse_zed_imu_data(sample_zed_imu_data)
        imu_processor._calibrate_imu(imu_data)
    
    assert imu_processor.is_calibrated
    assert len(imu_processor.calibration_samples) >= imu_processor.config['calibration_samples']

def test_gravity_compensation(imu_processor, sample_zed_imu_data):
    """Test gravity compensation"""
    # Set up calibrated processor
    imu_processor.is_calibrated = True
    imu_processor.gravity_vector = np.array([0, 0, -9.81])
    
    imu_data = imu_processor._parse_zed_imu_data(sample_zed_imu_data)
    compensated = imu_processor._compensate_gravity(imu_data)
    
    assert isinstance(compensated, IMUData)
    # Linear acceleration should be different from raw acceleration
    assert not np.array_equal(compensated.linear_acceleration, compensated.acceleration)

def test_imu_data_filtering(imu_processor, sample_zed_imu_data):
    """Test IMU data filtering"""
    # Add some history
    for _ in range(5):
        imu_data = imu_processor._parse_zed_imu_data(sample_zed_imu_data)
        imu_processor.imu_history.append(imu_data)
    
    # Filter new data
    new_imu_data = imu_processor._parse_zed_imu_data(sample_zed_imu_data)
    filtered = imu_processor._filter_imu_data(new_imu_data)
    
    assert isinstance(filtered, IMUData)
    # Filtered data should be smoothed
    assert not np.array_equal(filtered.acceleration, new_imu_data.acceleration)

def test_kalman_filter_initialization(imu_processor):
    """Test Kalman filter initialization"""
    kf = imu_processor.kalman_filter
    
    assert 'F' in kf  # State transition matrix
    assert 'Q' in kf  # Process noise
    assert 'R' in kf  # Measurement noise
    assert 'H' in kf  # Measurement matrix
    assert 'x' in kf  # State vector
    assert 'P' in kf  # Error covariance

def test_sensor_fusion(imu_processor, sample_zed_imu_data):
    """Test sensor fusion with Kalman filter"""
    imu_data = imu_processor._parse_zed_imu_data(sample_zed_imu_data)
    fused_data = imu_processor._sensor_fusion(imu_data)
    
    assert isinstance(fused_data, IMUData)
    # Fused orientation should be updated by Kalman filter
    assert len(fused_data.orientation) == 3

def test_motion_estimation(imu_processor, sample_zed_imu_data):
    """Test motion estimation"""
    # Set up calibrated processor
    imu_processor.is_calibrated = True
    
    imu_data = imu_processor._parse_zed_imu_data(sample_zed_imu_data)
    motion = imu_processor._estimate_motion(imu_data)
    
    assert isinstance(motion, VehicleMotion)
    assert len(motion.velocity) == 3
    assert len(motion.position) == 3
    assert len(motion.orientation) == 3

def test_motion_detection(imu_processor):
    """Test motion state detection"""
    # Create motion with high velocity
    motion = VehicleMotion(
        velocity=np.array([2.0, 0.0, 0.0]),  # 2 m/s
        position=np.zeros(3),
        orientation=np.zeros(3),
        angular_velocity=np.zeros(3),
        acceleration=np.array([1.0, 0.0, 0.0]),
        is_moving=False,
        motion_confidence=0.0
    )
    
    detected_motion = imu_processor._detect_motion_state(motion)
    
    assert detected_motion.is_moving
    assert detected_motion.motion_confidence > 0

def test_vehicle_heading(imu_processor):
    """Test vehicle heading calculation"""
    # Set current motion with known orientation
    imu_processor.current_motion.orientation = np.array([0, 0, np.pi/4])  # 45 degrees
    
    heading = imu_processor.get_vehicle_heading()
    assert abs(heading - np.pi/4) < 0.01

def test_vehicle_tilt(imu_processor):
    """Test vehicle tilt calculation"""
    # Set current motion with known orientation
    imu_processor.current_motion.orientation = np.array([0.1, 0.2, 0])  # Roll and pitch
    
    roll, pitch = imu_processor.get_vehicle_tilt()
    assert abs(roll - 0.1) < 0.01
    assert abs(pitch - 0.2) < 0.01

def test_stationary_detection(imu_processor):
    """Test stationary vehicle detection"""
    # Set low velocity
    imu_processor.current_motion.velocity = np.array([0.01, 0.01, 0.0])
    
    is_stationary = imu_processor.is_vehicle_stationary()
    assert is_stationary

def test_motion_summary(imu_processor):
    """Test motion summary generation"""
    # Set up some motion data
    imu_processor.current_motion = VehicleMotion(
        velocity=np.array([1.0, 0.5, 0.0]),
        position=np.array([10.0, 5.0, 0.0]),
        orientation=np.array([0.1, 0.2, 0.3]),
        angular_velocity=np.array([0.01, 0.02, 0.03]),
        acceleration=np.array([0.5, 0.0, 0.0]),
        is_moving=True,
        motion_confidence=0.8
    )
    imu_processor.is_calibrated = True
    
    summary = imu_processor.get_motion_summary()
    
    assert isinstance(summary, dict)
    assert 'is_moving' in summary
    assert 'velocity_magnitude' in summary
    assert 'speed_kmh' in summary
    assert 'heading_degrees' in summary
    assert 'roll_degrees' in summary
    assert 'pitch_degrees' in summary
    assert 'is_calibrated' in summary
    
    assert summary['is_moving']
    assert summary['is_calibrated']
    assert summary['speed_kmh'] > 0

def test_full_imu_processing_pipeline(imu_processor, sample_zed_imu_data):
    """Test complete IMU processing pipeline"""
    # Process multiple IMU samples to calibrate
    for _ in range(imu_processor.config['calibration_samples']):
        motion = imu_processor.process_imu_data(sample_zed_imu_data)
        assert isinstance(motion, VehicleMotion)
    
    # Should be calibrated now
    assert imu_processor.is_calibrated
    
    # Process more data
    for _ in range(10):
        motion = imu_processor.process_imu_data(sample_zed_imu_data)
        assert isinstance(motion, VehicleMotion)
        assert len(motion.velocity) == 3
        assert len(motion.position) == 3
    
    # Check history
    assert len(imu_processor.imu_history) > 0
    assert len(imu_processor.motion_history) > 0

def test_error_handling(imu_processor):
    """Test error handling in various scenarios"""
    # Test with invalid IMU data
    invalid_data = {}
    motion = imu_processor.process_imu_data(invalid_data)
    assert isinstance(motion, VehicleMotion)
    
    # Test with malformed data
    malformed_data = {'invalid_key': 'invalid_value'}
    motion = imu_processor.process_imu_data(malformed_data)
    assert isinstance(motion, VehicleMotion)

def test_performance_with_high_frequency_data(imu_processor, sample_zed_imu_data):
    """Test performance with high-frequency IMU data"""
    start_time = time.time()
    
    # Process 1000 samples (simulating 10 seconds at 100Hz)
    for _ in range(1000):
        imu_processor.process_imu_data(sample_zed_imu_data)
    
    end_time = time.time()
    processing_time = end_time - start_time
    
    # Should process 1000 samples in reasonable time (< 1 second)
    assert processing_time < 1.0
    
    # Check that history is properly limited
    assert len(imu_processor.imu_history) <= imu_processor.config['history_size']

def test_motion_confidence_calculation(imu_processor):
    """Test motion confidence calculation"""
    # Test with high motion
    high_motion = VehicleMotion(
        velocity=np.array([5.0, 0.0, 0.0]),
        position=np.zeros(3),
        orientation=np.zeros(3),
        angular_velocity=np.array([0.5, 0.0, 0.0]),
        acceleration=np.array([2.0, 0.0, 0.0]),
        is_moving=False,
        motion_confidence=0.0
    )
    
    detected = imu_processor._detect_motion_state(high_motion)
    assert detected.motion_confidence > 0.5
    
    # Test with low motion
    low_motion = VehicleMotion(
        velocity=np.array([0.01, 0.0, 0.0]),
        position=np.zeros(3),
        orientation=np.zeros(3),
        angular_velocity=np.array([0.001, 0.0, 0.0]),
        acceleration=np.array([0.01, 0.0, 0.0]),
        is_moving=False,
        motion_confidence=0.0
    )
    
    detected = imu_processor._detect_motion_state(low_motion)
    assert detected.motion_confidence < 0.5

def test_coordinate_system_consistency(imu_processor, sample_zed_imu_data):
    """Test coordinate system consistency"""
    # Process IMU data and check coordinate system
    motion = imu_processor.process_imu_data(sample_zed_imu_data)
    
    # Check that all vectors have 3 components
    assert len(motion.velocity) == 3
    assert len(motion.position) == 3
    assert len(motion.orientation) == 3
    assert len(motion.angular_velocity) == 3
    assert len(motion.acceleration) == 3
    
    # Check that orientation is in radians (reasonable range)
    for angle in motion.orientation:
        assert -2*np.pi <= angle <= 2*np.pi