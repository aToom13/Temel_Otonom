import pytest
from modules.road_processor import RoadProcessor

def test_road_processor_initialization():
    processor = RoadProcessor()
    assert processor is not None

def test_road_processor_process_road_no_obstacles_or_signs():
    processor = RoadProcessor()
    combined_data = {
        "detections": {"traffic_signs": []},
        "lanes": {"lanes": []},
        "obstacles": {"obstacle_detected": False}
    }
    direction_data = processor.process_road(combined_data)

    assert isinstance(direction_data, dict)
    assert direction_data["vehicle_status"] == "Düz"
    assert direction_data["target_speed"] == 30
    assert direction_data["steering_angle"] == 0

def test_road_processor_process_road_with_obstacle():
    processor = RoadProcessor()
    combined_data = {
        "detections": {"traffic_signs": []},
        "lanes": {"lanes": []},
        "obstacles": {"obstacle_detected": True, "distance": 500}
    }
    direction_data = processor.process_road(combined_data)

    assert direction_data["vehicle_status"] == "Dur"
    assert direction_data["target_speed"] == 0

def test_road_processor_process_road_with_stop_sign():
    processor = RoadProcessor()
    combined_data = {
        "detections": {"traffic_signs": [{"label": "Stop Sign", "confidence": 0.99, "bbox": []}]},
        "lanes": {"lanes": []},
        "obstacles": {"obstacle_detected": False}
    }
    direction_data = processor.process_road(combined_data)

    assert direction_data["vehicle_status"] == "Dur"
    assert direction_data["target_speed"] == 0
