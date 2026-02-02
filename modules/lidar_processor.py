"""
Slamtec RPLIDAR A1 2D sensör işleme modülü.
2D LiDAR verilerini işleme, obstacle detection ve mapping.
"""
import numpy as np
import time
import threading
import logging
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass
from collections import deque
import math
import os
import yaml

logger = logging.getLogger(__name__)

@dataclass
class LidarPoint:
    """LiDAR nokta verisi"""
    angle: float  # radians
    distance: float  # meters
    quality: int  # 0-255
    timestamp: float

@dataclass
class LidarScan:
    """Tam LiDAR tarama verisi"""
    points: List[LidarPoint]
    timestamp: float
    scan_frequency: float
    total_points: int

@dataclass
class DetectedObstacle:
    """Algılanan engel"""
    center_x: float
    center_y: float
    size: float
    confidence: float
    points_count: int

class RPLidarA1Processor:
    """
    Slamtec RPLIDAR A1 2D sensör işleme sınıfı.
    - Real-time LiDAR data processing
    - Obstacle detection ve clustering
    - 2D mapping ve visualization
    - Safety zone monitoring
    """
    
    def __init__(self, config_path: str = None):
        self.config = self._load_config(config_path)
        
        # LiDAR connection
        self.lidar = None
        self.is_connected = False
        self.is_scanning = False
        
        # Data processing
        self.scan_history = deque(maxlen=self.config.get('history_size', 10))
        self.current_scan = None
        
        # Obstacle detection
        self.obstacles = []
        self.safety_zones = self._init_safety_zones()
        
        # Threading
        self.scan_thread = None
        self.processing_thread = None
        self.running = False
        
        # Performance metrics
        self.scan_count = 0
        self.last_scan_time = 0
        self.scan_frequency = 0
        
        logger.info("RPLIDAR A1 Processor initialized")
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Konfigürasyon dosyasını yükle"""
        if config_path is None:
            config_path = os.path.join(os.path.dirname(__file__), '..', 'config.yaml')
        
        default_config = {
            'port': '/dev/ttyUSB0',  # Linux
            'baudrate': 115200,
            'timeout': 3.0,
            'max_distance': 12.0,  # meters (A1 max range)
            'min_distance': 0.15,  # meters (A1 min range)
            'angle_resolution': 1.0,  # degrees
            'history_size': 10,
            'clustering_distance': 0.3,  # meters
            'min_cluster_size': 3,
            'safety_zones': {
                'immediate': 0.5,  # meters
                'warning': 1.0,    # meters
                'caution': 2.0     # meters
            },
            'scan_frequency': 10.0,  # Hz
            'filter_noise': True,
            'median_filter_size': 3
        }
        
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)
                    lidar_config = config.get('lidar', {})
                    default_config.update(lidar_config)
            except Exception as e:
                logger.warning(f"Config loading failed: {e}, using defaults")
        
        return default_config
    
    def _init_safety_zones(self) -> Dict[str, float]:
        """Güvenlik bölgelerini başlat"""
        return self.config.get('safety_zones', {
            'immediate': 0.5,
            'warning': 1.0,
            'caution': 2.0
        })
    
    def connect(self) -> bool:
        """LiDAR'a bağlan"""
        try:
            # Try to import rplidar library
            try:
                from rplidar import RPLidar
            except ImportError:
                logger.error("rplidar library not found. Install with: pip install rplidar")
                return False
            
            # Connect to LiDAR
            port = self.config.get('port', '/dev/ttyUSB0')
            timeout = self.config.get('timeout', 3.0)
            
            self.lidar = RPLidar(port, timeout=timeout)
            
            # Test connection
            info = self.lidar.get_info()
            health = self.lidar.get_health()
            
            logger.info(f"LiDAR Info: {info}")
            logger.info(f"LiDAR Health: {health}")
            
            if health[0] == 'Good':
                self.is_connected = True
                logger.info("RPLIDAR A1 connected successfully")
                return True
            else:
                logger.error(f"LiDAR health check failed: {health}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to connect to RPLIDAR A1: {e}")
            self.lidar = None
            self.is_connected = False
            return False
    
    def disconnect(self):
        """LiDAR bağlantısını kes"""
        self.stop_scanning()
        
        if self.lidar:
            try:
                self.lidar.stop()
                self.lidar.stop_motor()
                self.lidar.disconnect()
            except Exception as e:
                logger.error(f"Error disconnecting LiDAR: {e}")
            finally:
                self.lidar = None
                self.is_connected = False
        
        logger.info("RPLIDAR A1 disconnected")
    
    def start_scanning(self) -> bool:
        """Taramayı başlat"""
        if not self.is_connected:
            logger.error("LiDAR not connected")
            return False
        
        if self.is_scanning:
            logger.warning("LiDAR already scanning")
            return True
        
        try:
            self.running = True
            self.is_scanning = True
            
            # Start motor
            self.lidar.start_motor()
            time.sleep(2)  # Wait for motor to stabilize
            
            # Start scanning thread
            self.scan_thread = threading.Thread(
                target=self._scan_loop,
                daemon=True,
                name="LiDAR-Scanner"
            )
            self.scan_thread.start()
            
            # Start processing thread
            self.processing_thread = threading.Thread(
                target=self._processing_loop,
                daemon=True,
                name="LiDAR-Processor"
            )
            self.processing_thread.start()
            
            logger.info("LiDAR scanning started")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start LiDAR scanning: {e}")
            self.is_scanning = False
            self.running = False
            return False
    
    def stop_scanning(self):
        """Taramayı durdur"""
        self.running = False
        self.is_scanning = False
        
        if self.lidar:
            try:
                self.lidar.stop()
                self.lidar.stop_motor()
            except Exception as e:
                logger.error(f"Error stopping LiDAR: {e}")
        
        # Wait for threads to finish
        if self.scan_thread and self.scan_thread.is_alive():
            self.scan_thread.join(timeout=2.0)
        
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=2.0)
        
        logger.info("LiDAR scanning stopped")
    
    def _scan_loop(self):
        """Ana tarama döngüsü"""
        try:
            for scan in self.lidar.iter_scans():
                if not self.running:
                    break
                
                # Process raw scan data
                processed_scan = self._process_raw_scan(scan)
                
                if processed_scan:
                    self.current_scan = processed_scan
                    self.scan_history.append(processed_scan)
                    
                    # Update performance metrics
                    self._update_scan_metrics()
                
                time.sleep(1.0 / self.config.get('scan_frequency', 10.0))
                
        except Exception as e:
            logger.error(f"LiDAR scan loop error: {e}")
            self.is_scanning = False
    
    def _processing_loop(self):
        """Ana işleme döngüsü"""
        while self.running:
            try:
                if self.current_scan:
                    # Obstacle detection
                    self.obstacles = self._detect_obstacles(self.current_scan)
                    
                    # Safety zone analysis
                    self._analyze_safety_zones()
                
                time.sleep(0.1)  # 10 Hz processing
                
            except Exception as e:
                logger.error(f"LiDAR processing loop error: {e}")
                time.sleep(0.5)
    
    def _process_raw_scan(self, raw_scan) -> Optional[LidarScan]:
        """Ham tarama verisini işle"""
        try:
            points = []
            timestamp = time.time()
            
            for quality, angle, distance in raw_scan:
                # Convert to standard units
                angle_rad = math.radians(angle)
                distance_m = distance / 1000.0  # mm to meters
                
                # Filter by distance range
                if (self.config['min_distance'] <= distance_m <= self.config['max_distance'] 
                    and quality > 0):
                    
                    point = LidarPoint(
                        angle=angle_rad,
                        distance=distance_m,
                        quality=quality,
                        timestamp=timestamp
                    )
                    points.append(point)
            
            if len(points) < 10:  # Minimum points threshold
                return None
            
            # Apply noise filtering
            if self.config.get('filter_noise', True):
                points = self._filter_noise(points)
            
            scan = LidarScan(
                points=points,
                timestamp=timestamp,
                scan_frequency=self.scan_frequency,
                total_points=len(points)
            )
            
            return scan
            
        except Exception as e:
            logger.error(f"Error processing raw scan: {e}")
            return None
    
    def _filter_noise(self, points: List[LidarPoint]) -> List[LidarPoint]:
        """Gürültü filtreleme"""
        if len(points) < 3:
            return points
        
        filtered_points = []
        filter_size = self.config.get('median_filter_size', 3)
        
        for i in range(len(points)):
            # Get neighboring points for median filtering
            start_idx = max(0, i - filter_size // 2)
            end_idx = min(len(points), i + filter_size // 2 + 1)
            
            neighbor_distances = [points[j].distance for j in range(start_idx, end_idx)]
            median_distance = np.median(neighbor_distances)
            
            # Keep point if it's close to median
            if abs(points[i].distance - median_distance) < 0.5:  # 50cm threshold
                filtered_points.append(points[i])
        
        return filtered_points
    
    def _detect_obstacles(self, scan: LidarScan) -> List[DetectedObstacle]:
        """Engel algılama"""
        if not scan.points:
            return []
        
        obstacles = []
        
        # Convert to Cartesian coordinates
        cartesian_points = []
        for point in scan.points:
            x = point.distance * math.cos(point.angle)
            y = point.distance * math.sin(point.angle)
            cartesian_points.append((x, y, point))
        
        # Simple clustering algorithm
        clusters = self._cluster_points(cartesian_points)
        
        # Convert clusters to obstacles
        for cluster in clusters:
            if len(cluster) >= self.config.get('min_cluster_size', 3):
                obstacle = self._cluster_to_obstacle(cluster)
                if obstacle:
                    obstacles.append(obstacle)
        
        return obstacles
    
    def _cluster_points(self, points: List[Tuple[float, float, LidarPoint]]) -> List[List[Tuple[float, float, LidarPoint]]]:
        """Noktaları kümelere ayır"""
        if not points:
            return []
        
        clusters = []
        used = set()
        clustering_distance = self.config.get('clustering_distance', 0.3)
        
        for i, (x1, y1, point1) in enumerate(points):
            if i in used:
                continue
            
            cluster = [(x1, y1, point1)]
            used.add(i)
            
            # Find nearby points
            for j, (x2, y2, point2) in enumerate(points):
                if j in used:
                    continue
                
                distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                if distance < clustering_distance:
                    cluster.append((x2, y2, point2))
                    used.add(j)
            
            if len(cluster) >= self.config.get('min_cluster_size', 3):
                clusters.append(cluster)
        
        return clusters
    
    def _cluster_to_obstacle(self, cluster: List[Tuple[float, float, LidarPoint]]) -> Optional[DetectedObstacle]:
        """Kümeyi engele çevir"""
        if not cluster:
            return None
        
        # Calculate center
        x_coords = [point[0] for point in cluster]
        y_coords = [point[1] for point in cluster]
        
        center_x = np.mean(x_coords)
        center_y = np.mean(y_coords)
        
        # Calculate size (max distance from center)
        max_distance = 0
        for x, y, _ in cluster:
            distance = math.sqrt((x - center_x)**2 + (y - center_y)**2)
            max_distance = max(max_distance, distance)
        
        # Calculate confidence based on point count and quality
        total_quality = sum(point[2].quality for point in cluster)
        avg_quality = total_quality / len(cluster)
        confidence = min(1.0, (len(cluster) / 10.0) * (avg_quality / 255.0))
        
        return DetectedObstacle(
            center_x=center_x,
            center_y=center_y,
            size=max_distance * 2,  # diameter
            confidence=confidence,
            points_count=len(cluster)
        )
    
    def _analyze_safety_zones(self):
        """Güvenlik bölgelerini analiz et"""
        if not self.obstacles:
            return
        
        zone_violations = {
            'immediate': [],
            'warning': [],
            'caution': []
        }
        
        for obstacle in self.obstacles:
            distance = math.sqrt(obstacle.center_x**2 + obstacle.center_y**2)
            
            if distance < self.safety_zones['immediate']:
                zone_violations['immediate'].append(obstacle)
            elif distance < self.safety_zones['warning']:
                zone_violations['warning'].append(obstacle)
            elif distance < self.safety_zones['caution']:
                zone_violations['caution'].append(obstacle)
        
        # Log violations
        if zone_violations['immediate']:
            logger.warning(f"IMMEDIATE DANGER: {len(zone_violations['immediate'])} obstacles detected")
        elif zone_violations['warning']:
            logger.warning(f"WARNING: {len(zone_violations['warning'])} obstacles in warning zone")
    
    def _update_scan_metrics(self):
        """Tarama metriklerini güncelle"""
        current_time = time.time()
        self.scan_count += 1
        
        if self.last_scan_time > 0:
            time_diff = current_time - self.last_scan_time
            if time_diff > 0:
                self.scan_frequency = 1.0 / time_diff
        
        self.last_scan_time = current_time
    
    def get_current_scan(self) -> Optional[LidarScan]:
        """Mevcut taramayı al"""
        return self.current_scan
    
    def get_obstacles(self) -> List[DetectedObstacle]:
        """Algılanan engelleri al"""
        return self.obstacles.copy()
    
    def get_scan_data_for_visualization(self) -> Dict[str, Any]:
        """Görselleştirme için tarama verisi"""
        if not self.current_scan:
            return {
                'points': [],
                'obstacles': [],
                'timestamp': 0,
                'scan_frequency': 0,
                'total_points': 0
            }
        
        # Convert points to visualization format
        viz_points = []
        for point in self.current_scan.points:
            x = point.distance * math.cos(point.angle)
            y = point.distance * math.sin(point.angle)
            viz_points.append({
                'x': x,
                'y': y,
                'distance': point.distance,
                'angle': math.degrees(point.angle),
                'quality': point.quality
            })
        
        # Convert obstacles to visualization format
        viz_obstacles = []
        for obstacle in self.obstacles:
            viz_obstacles.append({
                'center_x': obstacle.center_x,
                'center_y': obstacle.center_y,
                'size': obstacle.size,
                'confidence': obstacle.confidence,
                'points_count': obstacle.points_count
            })
        
        return {
            'points': viz_points,
            'obstacles': viz_obstacles,
            'timestamp': self.current_scan.timestamp,
            'scan_frequency': self.scan_frequency,
            'total_points': self.current_scan.total_points,
            'safety_zones': self.safety_zones,
            'is_scanning': self.is_scanning,
            'is_connected': self.is_connected
        }
    
    def get_status(self) -> Dict[str, Any]:
        """LiDAR durumunu al"""
        return {
            'is_connected': self.is_connected,
            'is_scanning': self.is_scanning,
            'scan_frequency': self.scan_frequency,
            'scan_count': self.scan_count,
            'obstacle_count': len(self.obstacles),
            'last_scan_time': self.last_scan_time,
            'config': self.config
        }