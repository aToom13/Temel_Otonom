import pytest
from modules.direction_controller import DirectionController

def test_direction_controller_initialization():
    controller = DirectionController()
    assert controller is not None

def test_direction_controller_control_returns_expected_data():
    controller = DirectionController()
    dummy_direction_data = {
        "steering_angle": 15,
        "target_speed": 20,
        "vehicle_status": "Viraj"
    }
    control_signals = controller.control(dummy_direction_data)

    assert isinstance(control_signals, dict)
    assert control_signals["angle"] == 15
    assert control_signals["speed"] == 20
    assert control_signals["status"] == "Viraj"
