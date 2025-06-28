import pytest
import numpy as np
import time
from unittest.mock import Mock, patch, MagicMock
from modules.enhanced_camera_manager import EnhancedCameraManager, CameraFrame, CameraStatus

class MockZEDCamera:
    """Mock ZED Camera for testing"""
    def __init__(self):
        self.is_opened = False
        
    def open(self, init_params):
        self.is_opened = True
        return MockErrorCode.SUCCESS
        
    def close(self):
        self.is_opened = False
        
    def grab(self, runtime_params):
        if self.is_opened:
            return MockErrorCode.SUCCESS
        return MockErrorCode.FAILURE
        
    def retrieve_image(self, image, view):
        # Return mock RGB image
        image.data = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
        
    def retrieve_measure(self, measure, measure_type):
        # Return mock depth/confidence data
        if "DEPTH" in str(measure_type):
            measure.data = np.random.rand(720, 1280).astype(np.float32) * 10
        else:  # Confidence
            measure.data = np.random.rand(720, 1280).astype(np.float32)
    
    def get_sensors_data(self, sensors_data, time_ref):
        # Mock IMU data
        return MockErrorCode.SUCCESS
        
    def get_camera_information(self):
        return MockCameraInfo()

class MockErrorCode:
    SUCCESS = 0
    FAILURE = 1

class MockCameraInfo:
    def __init__(self):
        self.camera_configuration = MockCameraConfig()

class MockCameraConfig:
    def __init__(self):
        self.resolution = MockResolution()

class MockResolution:
    def __init__(self):
        self.width = 1280
        self.height = 720

class MockIMUData:
    def __init__(self):
        self.linear_acceleration = [0.1, 0.2, 9.8]
        self.angular_velocity = [0.01, 0.02, 0.03]
        self.orientation = [0.0, 0.0, 0.0, 1.0]  # quaternion

class MockSensorsData:
    def get_imu_data(self):
        return MockIMUData()

@pytest.fixture
def mock_zed():
    """Mock ZED SDK"""
    with patch('modules.enhanced_camera_manager.sl') as mock_sl:
        mock_sl.Camera = MockZEDCamera
        mock_sl.ERROR_CODE = MockErrorCode
        mock_sl.RESOLUTION.HD720 = "HD720"
        mock_sl.DEPTH_MODE.PERFORMANCE = "PERFORMANCE"
        mock_sl.UNIT.METER = "METER"
        mock_sl.VIEW.LEFT = "LEFT"
        mock_sl.MEASURE.DEPTH = "DEPTH"
        mock_sl.MEASURE.CONFIDENCE = "CONFIDENCE"
        mock_sl.TIME_REFERENCE.IMAGE = "IMAGE"
        mock_sl.Mat = Mock
        mock_sl.InitParameters = Mock
        mock_sl.RuntimeParameters = Mock
        yield mock_sl

@pytest.fixture
def mock_cv2():
    """Mock OpenCV"""
    with patch('modules.enhanced_camera_manager.cv2') as mock_cv2:
        mock_cv2.VideoCapture.return_value.isOpened.return_value = True
        mock_cv2.VideoCapture.return_value.read.return_value = (True, np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8))
        mock_cv2.VideoCapture.return_value.get.return_value = 30.0
        mock_cv2.VideoCapture.return_value.set.return_value = True
        yield mock_cv2

@pytest.fixture
def camera_manager(mock_zed, mock_cv2):
    """Create camera manager with mocked dependencies"""
    return EnhancedCameraManager()

def test_camera_manager_initialization(camera_manager):
    """Test camera manager initialization"""
    assert camera_manager is not None
    assert camera_manager.current_camera_type == "None"
    assert not camera_manager.camera_status.is_connected
    assert camera_manager.imu_processor is not None

def test_zed_camera_initialization(camera_manager, mock_zed):
    """Test ZED camera initialization"""
    success = camera_manager._init_zed_camera()
    assert success
    assert camera_manager.zed_camera is not None
    assert camera_manager.camera_status.camera_type == "ZED"
    assert camera_manager.camera_status.has_depth
    assert camera_manager.camera_status.has_imu

def test_webcam_initialization(camera_manager, mock_cv2):
    """Test webcam initialization"""
    success = camera_manager._init_webcam()
    assert success
    assert camera_manager.webcam is not None
    assert camera_manager.camera_status.camera_type == "Webcam"
    assert not camera_manager.camera_status.has_depth
    assert not camera_manager.camera_status.has_imu

def test_camera_start_with_zed(camera_manager, mock_zed):
    """Test camera system start with ZED available"""
    success = camera_manager.start()
    assert success
    assert camera_manager.current_camera_type == "ZED"
    assert camera_manager.monitoring_active

def test_camera_start_fallback_to_webcam(camera_manager, mock_cv2):
    """Test camera system start with fallback to webcam"""
    # Mock ZED initialization failure
    with patch.object(camera_manager, '_init_zed_camera', return_value=False):
        success = camera_manager.start()
        assert success
        assert camera_manager.current_camera_type == "Webcam"

def test_camera_start_no_camera_available(camera_manager):
    """Test camera system start with no cameras available"""
    with patch.object(camera_manager, '_init_zed_camera', return_value=False):
        with patch.object(camera_manager, '_init_webcam', return_value=False):
            success = camera_manager.start()
            assert not success
            assert camera_manager.current_camera_type == "None"

def test_zed_frame_capture(camera_manager, mock_zed):
    """Test ZED frame capture"""
    camera_manager._init_zed_camera()
    camera_manager.current_camera_type = "ZED"
    
    frame = camera_manager._capture_zed_frame()
    assert frame is not None
    assert isinstance(frame, CameraFrame)
    assert frame.rgb is not None
    assert frame.depth is not None
    assert frame.confidence is not None
    assert frame.camera_type == "ZED"

def test_webcam_frame_capture(camera_manager, mock_cv2):
    """Test webcam frame capture"""
    camera_manager._init_webcam()
    camera_manager.current_camera_type = "Webcam"
    
    frame = camera_manager._capture_webcam_frame()
    assert frame is not None
    assert isinstance(frame, CameraFrame)
    assert frame.rgb is not None
    assert frame.depth is None
    assert frame.confidence is None
    assert frame.camera_type == "Webcam"

def test_camera_hot_swap(camera_manager, mock_zed, mock_cv2):
    """Test camera hot-swap functionality"""
    # Start with webcam
    camera_manager._init_webcam()
    camera_manager.current_camera_type = "Webcam"
    
    # Switch to ZED
    success = camera_manager.switch_to_zed_if_available()
    assert success
    assert camera_manager.current_camera_type == "ZED"
    assert camera_manager.webcam is None  # Webcam should be cleaned up

def test_camera_health_monitoring(camera_manager, mock_zed):
    """Test camera health monitoring"""
    camera_manager._init_zed_camera()
    camera_manager.current_camera_type = "ZED"
    
    # Test health check
    health = camera_manager._check_camera_health()
    assert health

def test_camera_reconnection(camera_manager, mock_zed, mock_cv2):
    """Test camera reconnection functionality"""
    camera_manager.config['auto_reconnect'] = True
    
    # Simulate camera failure and reconnection
    with patch.object(camera_manager, '_check_camera_health', return_value=False):
        camera_manager._attempt_reconnection()
        # Should attempt to reconnect

def test_imu_data_processing(camera_manager, mock_zed):
    """Test IMU data processing"""
    camera_manager._init_zed_camera()
    camera_manager.current_camera_type = "ZED"
    
    # Mock IMU data
    mock_imu_data = {
        'linear_acceleration': {'x': 0.1, 'y': 0.2, 'z': 9.8},
        'angular_velocity': {'x': 0.01, 'y': 0.02, 'z': 0.03},
        'orientation': {'x': 0.0, 'y': 0.0, 'z': 0.0, 'w': 1.0}
    }
    
    # Test IMU data retrieval
    imu_data = camera_manager.get_imu_data()
    assert isinstance(imu_data, dict)

def test_camera_status_reporting(camera_manager, mock_zed):
    """Test camera status reporting"""
    camera_manager._init_zed_camera()
    
    status = camera_manager.get_camera_status()
    assert isinstance(status, CameraStatus)
    assert status.is_connected
    assert status.camera_type == "ZED"
    assert status.has_depth
    assert status.has_imu

def test_fps_calculation(camera_manager, mock_zed):
    """Test FPS calculation"""
    camera_manager._init_zed_camera()
    
    # Simulate multiple frame captures
    for _ in range(5):
        camera_manager._update_fps()
        time.sleep(0.1)
    
    assert camera_manager.current_fps > 0

def test_camera_capabilities(camera_manager, mock_zed, mock_cv2):
    """Test camera capability detection"""
    # Test ZED capabilities
    camera_manager._init_zed_camera()
    camera_manager.current_camera_type = "ZED"
    
    assert camera_manager.has_depth_capability()
    assert camera_manager.has_imu_capability()
    
    # Test webcam capabilities
    camera_manager._init_webcam()
    camera_manager.current_camera_type = "Webcam"
    
    assert not camera_manager.has_depth_capability()
    assert not camera_manager.has_imu_capability()

def test_camera_cleanup(camera_manager, mock_zed, mock_cv2):
    """Test camera cleanup"""
    # Initialize both cameras
    camera_manager._init_zed_camera()
    camera_manager._init_webcam()
    
    # Cleanup
    camera_manager._cleanup_cameras()
    
    assert camera_manager.zed_camera is None
    assert camera_manager.webcam is None

def test_camera_stop(camera_manager, mock_zed):
    """Test camera system stop"""
    camera_manager.start()
    assert camera_manager.monitoring_active
    
    camera_manager.stop()
    assert not camera_manager.monitoring_active

def test_config_loading(camera_manager):
    """Test configuration loading"""
    config = camera_manager.config
    
    assert 'zed_resolution' in config
    assert 'zed_fps' in config
    assert 'fallback_webcam_index' in config
    assert 'auto_reconnect' in config

def test_error_handling(camera_manager):
    """Test error handling in various scenarios"""
    # Test frame capture without camera
    frame = camera_manager.capture_frame()
    assert frame is None
    
    # Test ZED initialization failure
    with patch('modules.enhanced_camera_manager.sl', side_effect=ImportError("pyzed not found")):
        success = camera_manager._init_zed_camera()
        assert not success
    
    # Test webcam initialization failure
    with patch('modules.enhanced_camera_manager.cv2.VideoCapture') as mock_cap:
        mock_cap.return_value.isOpened.return_value = False
        success = camera_manager._init_webcam()
        assert not success

def test_frame_data_structure(camera_manager, mock_zed):
    """Test frame data structure consistency"""
    camera_manager._init_zed_camera()
    camera_manager.current_camera_type = "ZED"
    
    frame = camera_manager._capture_zed_frame()
    
    assert hasattr(frame, 'rgb')
    assert hasattr(frame, 'depth')
    assert hasattr(frame, 'confidence')
    assert hasattr(frame, 'timestamp')
    assert hasattr(frame, 'camera_type')
    assert hasattr(frame, 'resolution')
    
    assert frame.timestamp > 0
    assert frame.resolution == (1280, 720)

# Integration test
def test_full_camera_pipeline(camera_manager, mock_zed):
    """Test complete camera processing pipeline"""
    # Start camera system
    assert camera_manager.start()
    
    # Capture frames
    for _ in range(3):
        frame = camera_manager.capture_frame()
        assert frame is not None
        assert frame.rgb is not None
        
        if camera_manager.has_imu_capability():
            imu_data = camera_manager.get_imu_data()
            assert isinstance(imu_data, dict)
    
    # Check status
    status = camera_manager.get_camera_status()
    assert status.is_connected
    
    # Stop system
    camera_manager.stop()
    assert not camera_manager.monitoring_active