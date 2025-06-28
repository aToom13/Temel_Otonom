import pytest
import numpy as np
import time
from unittest.mock import Mock, patch, MagicMock
from modules.lidar_processor import RPLidarA1Processor, LidarPoint, LidarScan, DetectedObstacle

class MockRPLidar:
    """Mock RPLidar class for testing"""
    def __init__(self, port, timeout=3.0):
        self.port = port
        self.timeout = timeout
        self.is_open = False
        self.motor_running = False
        
    def get_info(self):
        return {'model': 'A1', 'firmware': '1.29', 'hardware': '5'}
    
    def get_health(self):
        return ('Good', 'OK')
    
    def start_motor(self):
        self.motor_running = True
        
    def stop_motor(self):
        self.motor_running = False
        
    def stop(self):
        pass
        
    def disconnect(self):
        self.is_open = False
        
    def iter_scans(self):
        """Generate mock scan data"""
        while True:
            # Generate a circular pattern of points
            scan_data = []
            for angle in range(0, 360, 5):  # Every 5 degrees
                distance = 2000 + 500 * np.sin(np.radians(angle * 2))  # Varying distance
                quality = 200 + int(50 * np.random.random())  # Random quality
                scan_data.append((quality, angle, distance))
            
            yield scan_data
            time.sleep(0.1)  # 10 Hz

@pytest.fixture
def mock_rplidar():
    """Mock rplidar library"""
    with patch('modules.lidar_processor.RPLidar', MockRPLidar):
        yield

@pytest.fixture
def lidar_processor(mock_rplidar):
    """Create LiDAR processor with mocked hardware"""
    processor = RPLidarA1Processor()
    return processor

def test_lidar_processor_initialization(lidar_processor):
    """Test LiDAR processor initialization"""
    assert lidar_processor is not None
    assert not lidar_processor.is_connected
    assert not lidar_processor.is_scanning
    assert lidar_processor.config is not None

def test_lidar_processor_config_loading():
    """Test configuration loading"""
    processor = RPLidarA1Processor()
    config = processor.config
    
    assert 'port' in config
    assert 'max_distance' in config
    assert 'min_distance' in config
    assert 'safety_zones' in config
    assert config['max_distance'] == 12.0  # RPLIDAR A1 max range

def test_lidar_connection(lidar_processor):
    """Test LiDAR connection"""
    # Test successful connection
    success = lidar_processor.connect()
    assert success
    assert lidar_processor.is_connected
    
    # Test disconnection
    lidar_processor.disconnect()
    assert not lidar_processor.is_connected

def test_lidar_scanning(lidar_processor):
    """Test LiDAR scanning functionality"""
    # Connect first
    lidar_processor.connect()
    assert lidar_processor.is_connected
    
    # Start scanning
    success = lidar_processor.start_scanning()
    assert success
    assert lidar_processor.is_scanning
    
    # Wait for some scans
    time.sleep(0.5)
    
    # Check if we have scan data
    current_scan = lidar_processor.get_current_scan()
    assert current_scan is not None
    assert isinstance(current_scan, LidarScan)
    assert len(current_scan.points) > 0
    
    # Stop scanning
    lidar_processor.stop_scanning()
    assert not lidar_processor.is_scanning

def test_lidar_point_processing(lidar_processor):
    """Test LiDAR point processing"""
    # Create mock raw scan data
    raw_scan = [
        (200, 0, 1000),    # quality, angle, distance (mm)
        (180, 90, 2000),
        (220, 180, 1500),
        (190, 270, 3000),
    ]
    
    # Process raw scan
    processed_scan = lidar_processor._process_raw_scan(raw_scan)
    
    assert processed_scan is not None
    assert isinstance(processed_scan, LidarScan)
    assert len(processed_scan.points) == 4
    
    # Check point conversion
    point = processed_scan.points[0]
    assert isinstance(point, LidarPoint)
    assert point.distance == 1.0  # 1000mm = 1.0m
    assert point.angle == 0.0
    assert point.quality == 200

def test_obstacle_detection(lidar_processor):
    """Test obstacle detection algorithm"""
    # Create a scan with clustered points (simulating an obstacle)
    points = []
    
    # Add clustered points (obstacle at 2m, 45 degrees)
    for i in range(10):
        angle = np.radians(45 + i * 2)  # Small angle spread
        distance = 2.0 + 0.1 * np.random.random()  # Small distance variation
        point = LidarPoint(
            angle=angle,
            distance=distance,
            quality=200,
            timestamp=time.time()
        )
        points.append(point)
    
    # Add scattered points (background)
    for i in range(20):
        angle = np.radians(np.random.random() * 360)
        distance = 5.0 + 2.0 * np.random.random()
        point = LidarPoint(
            angle=angle,
            distance=distance,
            quality=150,
            timestamp=time.time()
        )
        points.append(point)
    
    scan = LidarScan(
        points=points,
        timestamp=time.time(),
        scan_frequency=10.0,
        total_points=len(points)
    )
    
    # Detect obstacles
    obstacles = lidar_processor._detect_obstacles(scan)
    
    assert len(obstacles) >= 1  # Should detect at least one obstacle
    
    # Check obstacle properties
    obstacle = obstacles[0]
    assert isinstance(obstacle, DetectedObstacle)
    assert obstacle.confidence > 0
    assert obstacle.points_count >= lidar_processor.config['min_cluster_size']

def test_safety_zone_analysis(lidar_processor):
    """Test safety zone analysis"""
    # Create obstacles at different distances
    lidar_processor.obstacles = [
        DetectedObstacle(
            center_x=0.3, center_y=0.0,  # 0.3m away (immediate zone)
            size=0.2, confidence=0.9, points_count=10
        ),
        DetectedObstacle(
            center_x=0.8, center_y=0.0,  # 0.8m away (warning zone)
            size=0.3, confidence=0.8, points_count=8
        ),
        DetectedObstacle(
            center_x=1.5, center_y=0.0,  # 1.5m away (caution zone)
            size=0.4, confidence=0.7, points_count=6
        ),
    ]
    
    # Analyze safety zones (this should log warnings)
    lidar_processor._analyze_safety_zones()
    
    # Test passes if no exceptions are raised

def test_noise_filtering(lidar_processor):
    """Test noise filtering functionality"""
    # Create points with noise
    points = []
    for i in range(20):
        # Add normal points
        point = LidarPoint(
            angle=np.radians(i * 10),
            distance=2.0,
            quality=200,
            timestamp=time.time()
        )
        points.append(point)
        
        # Add noise point
        if i % 5 == 0:
            noise_point = LidarPoint(
                angle=np.radians(i * 10 + 1),
                distance=10.0,  # Far outlier
                quality=50,     # Low quality
                timestamp=time.time()
            )
            points.append(noise_point)
    
    # Apply noise filtering
    filtered_points = lidar_processor._filter_noise(points)
    
    # Should have fewer points after filtering
    assert len(filtered_points) <= len(points)
    
    # Filtered points should have reasonable distances
    for point in filtered_points:
        assert point.distance < 8.0  # Most noise should be removed

def test_clustering_algorithm(lidar_processor):
    """Test point clustering algorithm"""
    # Create cartesian points for clustering
    cartesian_points = []
    
    # Cluster 1: Points around (1, 1)
    for i in range(5):
        x = 1.0 + 0.1 * np.random.random()
        y = 1.0 + 0.1 * np.random.random()
        point = LidarPoint(0, 0, 200, time.time())
        cartesian_points.append((x, y, point))
    
    # Cluster 2: Points around (3, 2)
    for i in range(5):
        x = 3.0 + 0.1 * np.random.random()
        y = 2.0 + 0.1 * np.random.random()
        point = LidarPoint(0, 0, 200, time.time())
        cartesian_points.append((x, y, point))
    
    # Isolated points
    for i in range(3):
        x = 5.0 + i
        y = 5.0 + i
        point = LidarPoint(0, 0, 200, time.time())
        cartesian_points.append((x, y, point))
    
    # Perform clustering
    clusters = lidar_processor._cluster_points(cartesian_points)
    
    # Should find 2 main clusters
    assert len(clusters) >= 2
    
    # Each cluster should have minimum required points
    for cluster in clusters:
        assert len(cluster) >= lidar_processor.config['min_cluster_size']

def test_visualization_data_format(lidar_processor):
    """Test visualization data format"""
    # Connect and start scanning
    lidar_processor.connect()
    lidar_processor.start_scanning()
    
    # Wait for some data
    time.sleep(0.3)
    
    # Get visualization data
    viz_data = lidar_processor.get_scan_data_for_visualization()
    
    # Check data structure
    assert isinstance(viz_data, dict)
    assert 'points' in viz_data
    assert 'obstacles' in viz_data
    assert 'timestamp' in viz_data
    assert 'scan_frequency' in viz_data
    assert 'total_points' in viz_data
    assert 'safety_zones' in viz_data
    assert 'is_scanning' in viz_data
    assert 'is_connected' in viz_data
    
    # Check points format
    if viz_data['points']:
        point = viz_data['points'][0]
        assert 'x' in point
        assert 'y' in point
        assert 'distance' in point
        assert 'angle' in point
        assert 'quality' in point
    
    # Stop scanning
    lidar_processor.stop_scanning()

def test_status_reporting(lidar_processor):
    """Test status reporting functionality"""
    # Get initial status
    status = lidar_processor.get_status()
    
    assert isinstance(status, dict)
    assert 'is_connected' in status
    assert 'is_scanning' in status
    assert 'scan_frequency' in status
    assert 'scan_count' in status
    assert 'obstacle_count' in status
    assert 'config' in status
    
    # Initially not connected
    assert not status['is_connected']
    assert not status['is_scanning']
    
    # Connect and check status
    lidar_processor.connect()
    status = lidar_processor.get_status()
    assert status['is_connected']

def test_performance_metrics(lidar_processor):
    """Test performance metrics tracking"""
    # Connect and start scanning
    lidar_processor.connect()
    lidar_processor.start_scanning()
    
    # Wait for multiple scans
    time.sleep(0.5)
    
    # Check metrics
    assert lidar_processor.scan_count > 0
    assert lidar_processor.scan_frequency > 0
    assert lidar_processor.last_scan_time > 0
    
    # Stop scanning
    lidar_processor.stop_scanning()

def test_error_handling(lidar_processor):
    """Test error handling in various scenarios"""
    # Test connection without hardware
    with patch('modules.lidar_processor.RPLidar', side_effect=Exception("Hardware not found")):
        processor = RPLidarA1Processor()
        success = processor.connect()
        assert not success
        assert not processor.is_connected
    
    # Test scanning without connection
    success = lidar_processor.start_scanning()
    assert not success
    
    # Test processing with empty scan
    result = lidar_processor._process_raw_scan([])
    assert result is None
    
    # Test obstacle detection with empty scan
    empty_scan = LidarScan([], time.time(), 0, 0)
    obstacles = lidar_processor._detect_obstacles(empty_scan)
    assert len(obstacles) == 0

def test_configuration_validation():
    """Test configuration validation"""
    # Test with custom config
    custom_config = {
        'max_distance': 8.0,
        'min_distance': 0.2,
        'clustering_distance': 0.5,
        'safety_zones': {
            'immediate': 0.3,
            'warning': 0.8,
            'caution': 1.5
        }
    }
    
    with patch('os.path.exists', return_value=True):
        with patch('builtins.open', mock_open_config(custom_config)):
            processor = RPLidarA1Processor()
            
            assert processor.config['max_distance'] == 8.0
            assert processor.config['min_distance'] == 0.2
            assert processor.config['clustering_distance'] == 0.5

def mock_open_config(config_data):
    """Mock file open for config testing"""
    import yaml
    from unittest.mock import mock_open
    
    yaml_content = yaml.dump({'lidar': config_data})
    return mock_open(read_data=yaml_content)

# Integration test
def test_full_lidar_pipeline(lidar_processor):
    """Test complete LiDAR processing pipeline"""
    # Connect
    assert lidar_processor.connect()
    
    # Start scanning
    assert lidar_processor.start_scanning()
    
    # Wait for data
    time.sleep(0.5)
    
    # Check all components are working
    scan = lidar_processor.get_current_scan()
    assert scan is not None
    
    obstacles = lidar_processor.get_obstacles()
    assert isinstance(obstacles, list)
    
    viz_data = lidar_processor.get_scan_data_for_visualization()
    assert viz_data['is_scanning']
    assert viz_data['is_connected']
    
    status = lidar_processor.get_status()
    assert status['is_connected']
    assert status['is_scanning']
    assert status['scan_count'] > 0
    
    # Cleanup
    lidar_processor.stop_scanning()
    lidar_processor.disconnect()
    
    # Verify cleanup
    assert not lidar_processor.is_scanning
    assert not lidar_processor.is_connected