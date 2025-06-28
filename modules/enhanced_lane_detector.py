"""
Gelişmiş şerit algılama modülü.
PRD'de belirtilen yapay zeka iyileştirmeleri için.
"""
import cv2
import numpy as np
from ultralytics import YOLO
from collections import deque
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
import logging
import os
import yaml

logger = logging.getLogger(__name__)

@dataclass
class LanePoint:
    """Şerit noktası"""
    x: float
    y: float
    confidence: float

@dataclass
class Lane:
    """Şerit sınıfı"""
    points: List[LanePoint]
    polynomial_coeffs: Optional[np.ndarray]
    lane_type: str  # 'left', 'right', 'center'
    confidence: float
    curvature: float
    width: Optional[float] = None

@dataclass
class LaneDetectionResult:
    """Şerit algılama sonucu"""
    lanes: List[Lane]
    lane_center_offset: float
    lane_departure_warning: bool
    lane_change_detected: bool
    road_curvature: float
    detection_quality: float

class EnhancedLaneDetector:
    """
    Gelişmiş şerit algılama sınıfı.
    - Temporal consistency
    - Multi-frame averaging
    - Curve prediction
    - Lane change detection
    - Advanced polynomial fitting
    """
    
    def __init__(self, model_path: str = None, config_path: str = None):
        # Konfigürasyon yükle
        self.config = self._load_config(config_path)
        
        # Model yükle
        self.model = self._load_model(model_path)
        
        # Temporal tracking
        self.lane_history = deque(maxsize=self.config.get('history_size', 10))
        self.frame_count = 0
        
        # Lane tracking
        self.previous_lanes = []
        self.lane_tracker = LaneTracker()
        
        # Detection parameters
        self.confidence_threshold = self.config.get('confidence_threshold', 0.5)
        self.temporal_smoothing = self.config.get('temporal_smoothing', 0.3)
        
        # Lane departure detection
        self.lane_departure_threshold = self.config.get('lane_departure_threshold', 0.3)
        self.lane_change_threshold = self.config.get('lane_change_threshold', 0.5)
        
        logger.info("Enhanced Lane Detector initialized")
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Konfigürasyon dosyasını yükle"""
        if config_path is None:
            config_path = os.path.join(os.path.dirname(__file__), '..', 'config.yaml')
        
        default_config = {
            'history_size': 10,
            'confidence_threshold': 0.5,
            'temporal_smoothing': 0.3,
            'lane_departure_threshold': 0.3,
            'lane_change_threshold': 0.5,
            'polynomial_degree': 2,
            'roi_top': 0.6,
            'roi_bottom': 0.95
        }
        
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)
                    lane_config = config.get('lane_detection', {})
                    default_config.update(lane_config)
            except Exception as e:
                logger.warning(f"Config loading failed: {e}, using defaults")
        
        return default_config
    
    def _load_model(self, model_path: str) -> Optional[YOLO]:
        """YOLO modelini yükle"""
        if model_path is None:
            model_path = self.config.get('model_path', 'models/serit.pt')
        
        if not os.path.exists(model_path):
            logger.warning(f"Lane model not found at {model_path}")
            try:
                model = YOLO('yolov8n-seg.pt')
                logger.info("Using default YOLOv8n-seg model")
                return model
            except Exception as e:
                logger.error(f"Failed to load fallback model: {e}")
                return None
        
        try:
            model = YOLO(model_path)
            logger.info(f"Lane detection model loaded from {model_path}")
            return model
        except Exception as e:
            logger.error(f"Failed to load lane model: {e}")
            return None
    
    def detect_lanes_with_tracking(self, frame: np.ndarray) -> LaneDetectionResult:
        """
        Gelişmiş şerit algılama ana fonksiyonu.
        
        Args:
            frame: Giriş görüntüsü
            
        Returns:
            Detaylı şerit algılama sonucu
        """
        self.frame_count += 1
        
        try:
            # 1. ROI extraction
            roi_frame = self._extract_roi(frame)
            
            # 2. Lane detection
            raw_lanes = self._detect_raw_lanes(roi_frame)
            
            # 3. Temporal consistency
            consistent_lanes = self._apply_temporal_consistency(raw_lanes)
            
            # 4. Lane tracking
            tracked_lanes = self.lane_tracker.update_tracks(consistent_lanes)
            
            # 5. Polynomial fitting
            fitted_lanes = self._fit_lane_polynomials(tracked_lanes, frame.shape)
            
            # 6. Lane analysis
            analysis_result = self._analyze_lanes(fitted_lanes, frame.shape)
            
            # 7. Update history
            self.lane_history.append(fitted_lanes)
            self.previous_lanes = fitted_lanes
            
            return analysis_result
            
        except Exception as e:
            logger.error(f"Lane detection failed: {e}")
            return LaneDetectionResult(
                lanes=[],
                lane_center_offset=0.0,
                lane_departure_warning=False,
                lane_change_detected=False,
                road_curvature=0.0,
                detection_quality=0.0
            )
    
    def _extract_roi(self, frame: np.ndarray) -> np.ndarray:
        """İlgi alanını çıkar"""
        h, w = frame.shape[:2]
        
        # ROI coordinates
        top = int(h * self.config['roi_top'])
        bottom = int(h * self.config['roi_bottom'])
        
        # Create mask
        mask = np.zeros((h, w), dtype=np.uint8)
        
        # Trapezoid ROI
        roi_points = np.array([
            [int(w * 0.1), bottom],
            [int(w * 0.4), top],
            [int(w * 0.6), top],
            [int(w * 0.9), bottom]
        ], dtype=np.int32)
        
        cv2.fillPoly(mask, [roi_points], 255)
        
        # Apply mask
        roi_frame = cv2.bitwise_and(frame, frame, mask=mask)
        
        return roi_frame
    
    def _detect_raw_lanes(self, frame: np.ndarray) -> List[Lane]:
        """Ham şerit algılama"""
        if self.model is None:
            return []
        
        try:
            # YOLO inference
            results = self.model(frame, verbose=False)
            
            lanes = []
            for r in results:
                if r.masks is not None:
                    for i, mask_data in enumerate(r.masks.data):
                        # Process mask
                        mask = mask_data.cpu().numpy()
                        mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
                        mask = (mask > 0.5).astype(np.uint8) * 255
                        
                        # Extract lane points
                        lane_points = self._extract_lane_points(mask)
                        
                        if len(lane_points) > 10:  # Minimum points threshold
                            # Determine lane type
                            lane_type = self._classify_lane_type(lane_points, frame.shape)
                            
                            # Calculate confidence
                            confidence = float(np.mean([p.confidence for p in lane_points]))
                            
                            lane = Lane(
                                points=lane_points,
                                polynomial_coeffs=None,
                                lane_type=lane_type,
                                confidence=confidence,
                                curvature=0.0
                            )
                            
                            lanes.append(lane)
            
            return lanes
            
        except Exception as e:
            logger.error(f"Raw lane detection failed: {e}")
            return []
    
    def _extract_lane_points(self, mask: np.ndarray) -> List[LanePoint]:
        """Maskeden şerit noktalarını çıkar"""
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return []
        
        # Get largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Extract points with confidence
        points = []
        for point in largest_contour.reshape(-1, 2):
            x, y = point
            # Simple confidence based on mask intensity
            confidence = float(mask[y, x]) / 255.0
            
            points.append(LanePoint(x=float(x), y=float(y), confidence=confidence))
        
        return points
    
    def _classify_lane_type(self, points: List[LanePoint], frame_shape: Tuple[int, int]) -> str:
        """Şerit tipini sınıflandır"""
        if not points:
            return 'unknown'
        
        # Calculate average x position
        avg_x = np.mean([p.x for p in points])
        frame_width = frame_shape[1]
        
        # Simple classification based on position
        if avg_x < frame_width * 0.4:
            return 'left'
        elif avg_x > frame_width * 0.6:
            return 'right'
        else:
            return 'center'
    
    def _apply_temporal_consistency(self, current_lanes: List[Lane]) -> List[Lane]:
        """Zamansal tutarlılık uygula"""
        if not self.previous_lanes:
            return current_lanes
        
        consistent_lanes = []
        
        for current_lane in current_lanes:
            # Find best matching previous lane
            best_match = self._find_best_lane_match(current_lane, self.previous_lanes)
            
            if best_match is not None:
                # Apply temporal smoothing
                smoothed_lane = self._smooth_lane_temporally(current_lane, best_match)
                consistent_lanes.append(smoothed_lane)
            else:
                consistent_lanes.append(current_lane)
        
        return consistent_lanes
    
    def _find_best_lane_match(self, current_lane: Lane, previous_lanes: List[Lane]) -> Optional[Lane]:
        """En iyi eşleşen önceki şeridi bul"""
        if not previous_lanes:
            return None
        
        best_match = None
        best_score = float('inf')
        
        for prev_lane in previous_lanes:
            if prev_lane.lane_type == current_lane.lane_type:
                # Calculate similarity score
                score = self._calculate_lane_similarity(current_lane, prev_lane)
                
                if score < best_score:
                    best_score = score
                    best_match = prev_lane
        
        return best_match if best_score < 100.0 else None  # Threshold
    
    def _calculate_lane_similarity(self, lane1: Lane, lane2: Lane) -> float:
        """İki şerit arasındaki benzerlik skorunu hesapla"""
        if not lane1.points or not lane2.points:
            return float('inf')
        
        # Simple distance-based similarity
        points1 = np.array([[p.x, p.y] for p in lane1.points])
        points2 = np.array([[p.x, p.y] for p in lane2.points])
        
        # Calculate average distance between point sets
        min_len = min(len(points1), len(points2))
        if min_len == 0:
            return float('inf')
        
        # Sample points for comparison
        indices1 = np.linspace(0, len(points1)-1, min_len, dtype=int)
        indices2 = np.linspace(0, len(points2)-1, min_len, dtype=int)
        
        sampled_points1 = points1[indices1]
        sampled_points2 = points2[indices2]
        
        distances = np.linalg.norm(sampled_points1 - sampled_points2, axis=1)
        return float(np.mean(distances))
    
    def _smooth_lane_temporally(self, current_lane: Lane, previous_lane: Lane) -> Lane:
        """Şeridi zamansal olarak yumuşat"""
        alpha = self.temporal_smoothing
        
        # Smooth points
        smoothed_points = []
        
        min_len = min(len(current_lane.points), len(previous_lane.points))
        
        for i in range(min_len):
            curr_p = current_lane.points[i]
            prev_p = previous_lane.points[i]
            
            smoothed_x = alpha * curr_p.x + (1 - alpha) * prev_p.x
            smoothed_y = alpha * curr_p.y + (1 - alpha) * prev_p.y
            smoothed_conf = max(curr_p.confidence, prev_p.confidence)
            
            smoothed_points.append(LanePoint(
                x=smoothed_x,
                y=smoothed_y,
                confidence=smoothed_conf
            ))
        
        # Add remaining points from current lane
        if len(current_lane.points) > min_len:
            smoothed_points.extend(current_lane.points[min_len:])
        
        return Lane(
            points=smoothed_points,
            polynomial_coeffs=current_lane.polynomial_coeffs,
            lane_type=current_lane.lane_type,
            confidence=current_lane.confidence,
            curvature=current_lane.curvature
        )
    
    def _fit_lane_polynomials(self, lanes: List[Lane], frame_shape: Tuple[int, int]) -> List[Lane]:
        """Şeritler için polinom uydur"""
        fitted_lanes = []
        
        for lane in lanes:
            if len(lane.points) < 3:  # Minimum points for polynomial
                fitted_lanes.append(lane)
                continue
            
            try:
                # Extract coordinates
                x_coords = np.array([p.x for p in lane.points])
                y_coords = np.array([p.y for p in lane.points])
                
                # Fit polynomial
                degree = self.config['polynomial_degree']
                coeffs = np.polyfit(y_coords, x_coords, degree)
                
                # Calculate curvature
                curvature = self._calculate_curvature(coeffs, frame_shape[0])
                
                # Update lane
                lane.polynomial_coeffs = coeffs
                lane.curvature = curvature
                
                fitted_lanes.append(lane)
                
            except Exception as e:
                logger.warning(f"Polynomial fitting failed for lane: {e}")
                fitted_lanes.append(lane)
        
        return fitted_lanes
    
    def _calculate_curvature(self, coeffs: np.ndarray, frame_height: int) -> float:
        """Polinom katsayılarından eğrilik hesapla"""
        if len(coeffs) < 3:
            return 0.0
        
        # Calculate curvature at bottom of frame
        y = frame_height - 1
        
        # First and second derivatives
        dy_dx = coeffs[-2] + 2 * coeffs[-3] * y
        d2y_dx2 = 2 * coeffs[-3]
        
        # Curvature formula
        curvature = abs(d2y_dx2) / (1 + dy_dx**2)**1.5
        
        return float(curvature)
    
    def _analyze_lanes(self, lanes: List[Lane], frame_shape: Tuple[int, int]) -> LaneDetectionResult:
        """Şerit analizi yap"""
        h, w = frame_shape[:2]
        
        # Find left and right lanes
        left_lanes = [l for l in lanes if l.lane_type == 'left']
        right_lanes = [l for l in lanes if l.lane_type == 'right']
        
        # Calculate lane center offset
        lane_center_offset = self._calculate_lane_center_offset(
            left_lanes, right_lanes, w
        )
        
        # Lane departure warning
        lane_departure_warning = abs(lane_center_offset) > self.lane_departure_threshold
        
        # Lane change detection
        lane_change_detected = self._detect_lane_change()
        
        # Road curvature
        road_curvature = self._calculate_road_curvature(lanes)
        
        # Detection quality
        detection_quality = self._assess_detection_quality(lanes)
        
        return LaneDetectionResult(
            lanes=lanes,
            lane_center_offset=lane_center_offset,
            lane_departure_warning=lane_departure_warning,
            lane_change_detected=lane_change_detected,
            road_curvature=road_curvature,
            detection_quality=detection_quality
        )
    
    def _calculate_lane_center_offset(self, 
                                    left_lanes: List[Lane], 
                                    right_lanes: List[Lane], 
                                    frame_width: int) -> float:
        """Şerit merkez sapmasını hesapla"""
        if not left_lanes and not right_lanes:
            return 0.0
        
        frame_center = frame_width / 2
        
        if left_lanes and right_lanes:
            # Both lanes available
            left_lane = max(left_lanes, key=lambda l: l.confidence)
            right_lane = max(right_lanes, key=lambda l: l.confidence)
            
            # Calculate lane center at bottom of frame
            if left_lane.polynomial_coeffs is not None and right_lane.polynomial_coeffs is not None:
                y_bottom = 480  # Assume 480p height
                left_x = np.polyval(left_lane.polynomial_coeffs, y_bottom)
                right_x = np.polyval(right_lane.polynomial_coeffs, y_bottom)
                
                lane_center = (left_x + right_x) / 2
                offset = (lane_center - frame_center) / frame_width
                
                return float(offset)
        
        elif left_lanes:
            # Only left lane
            left_lane = max(left_lanes, key=lambda l: l.confidence)
            if left_lane.polynomial_coeffs is not None:
                y_bottom = 480
                left_x = np.polyval(left_lane.polynomial_coeffs, y_bottom)
                # Assume standard lane width
                estimated_center = left_x + 200  # pixels
                offset = (estimated_center - frame_center) / frame_width
                return float(offset)
        
        elif right_lanes:
            # Only right lane
            right_lane = max(right_lanes, key=lambda l: l.confidence)
            if right_lane.polynomial_coeffs is not None:
                y_bottom = 480
                right_x = np.polyval(right_lane.polynomial_coeffs, y_bottom)
                # Assume standard lane width
                estimated_center = right_x - 200  # pixels
                offset = (estimated_center - frame_center) / frame_width
                return float(offset)
        
        return 0.0
    
    def _detect_lane_change(self) -> bool:
        """Şerit değişikliği algıla"""
        if len(self.lane_history) < 5:
            return False
        
        # Analyze lateral movement over time
        recent_offsets = []
        for lanes in list(self.lane_history)[-5:]:
            # Calculate center offset for each frame
            left_lanes = [l for l in lanes if l.lane_type == 'left']
            right_lanes = [l for l in lanes if l.lane_type == 'right']
            offset = self._calculate_lane_center_offset(left_lanes, right_lanes, 640)
            recent_offsets.append(offset)
        
        if len(recent_offsets) < 5:
            return False
        
        # Check for consistent lateral movement
        offset_change = recent_offsets[-1] - recent_offsets[0]
        
        return abs(offset_change) > self.lane_change_threshold
    
    def _calculate_road_curvature(self, lanes: List[Lane]) -> float:
        """Yol eğriliğini hesapla"""
        if not lanes:
            return 0.0
        
        # Average curvature of all lanes
        curvatures = [l.curvature for l in lanes if l.curvature is not None]
        
        if not curvatures:
            return 0.0
        
        return float(np.mean(curvatures))
    
    def _assess_detection_quality(self, lanes: List[Lane]) -> float:
        """Algılama kalitesini değerlendir"""
        if not lanes:
            return 0.0
        
        # Quality based on number of lanes and their confidence
        total_confidence = sum(l.confidence for l in lanes)
        lane_count_factor = min(1.0, len(lanes) / 2.0)  # Ideal: 2 lanes
        
        quality = (total_confidence / len(lanes)) * lane_count_factor
        
        return min(1.0, quality)

class LaneTracker:
    """Şerit takip sınıfı"""
    
    def __init__(self, max_age: int = 5):
        self.tracks = {}
        self.next_id = 0
        self.max_age = max_age
    
    def update_tracks(self, detected_lanes: List[Lane]) -> List[Lane]:
        """Şerit takiplerini güncelle"""
        # Age existing tracks
        for track_id in list(self.tracks.keys()):
            self.tracks[track_id]['age'] += 1
            if self.tracks[track_id]['age'] > self.max_age:
                del self.tracks[track_id]
        
        # Match detected lanes to tracks
        matched_lanes = []
        
        for lane in detected_lanes:
            best_track_id = self._find_best_track_match(lane)
            
            if best_track_id is not None:
                # Update existing track
                self.tracks[best_track_id] = {
                    'lane': lane,
                    'age': 0
                }
            else:
                # Create new track
                self.tracks[self.next_id] = {
                    'lane': lane,
                    'age': 0
                }
                self.next_id += 1
            
            matched_lanes.append(lane)
        
        return matched_lanes
    
    def _find_best_track_match(self, lane: Lane) -> Optional[int]:
        """En iyi eşleşen takibi bul"""
        best_track_id = None
        best_score = float('inf')
        
        for track_id, track in self.tracks.items():
            if track['lane'].lane_type == lane.lane_type:
                # Calculate similarity
                score = self._calculate_track_similarity(lane, track['lane'])
                
                if score < best_score and score < 50.0:  # Threshold
                    best_score = score
                    best_track_id = track_id
        
        return best_track_id
    
    def _calculate_track_similarity(self, lane1: Lane, lane2: Lane) -> float:
        """İki şerit arasındaki benzerlik skorunu hesapla"""
        if not lane1.points or not lane2.points:
            return float('inf')
        
        # Simple average position difference
        avg_x1 = np.mean([p.x for p in lane1.points])
        avg_x2 = np.mean([p.x for p in lane2.points])
        
        return abs(avg_x1 - avg_x2)