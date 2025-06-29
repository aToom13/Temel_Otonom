import pytest
import threading
import time
import sys
import os

# Add the project root to the sys.path to allow imports from modules and main
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import main as processing_main
from web_interface import app as flask_app

# Mock serial.Serial for integration tests to prevent actual hardware connection
class MockSerialIntegration:
    def __init__(self, port, baud_rate, timeout):
        self.port = port
        self.baud_rate = baud_rate
        self.timeout = timeout
        self.is_open = True
        self.written_data = []
        self.in_waiting = 0

    def close(self):
        self.is_open = False

    def write(self, data):
        self.written_data.append(data)
        return len(data)

    def readline(self):
        return b""

    def reset_input_buffer(self):
        pass

    def reset_output_buffer(self):
        pass

@pytest.fixture
def setup_integration_environment(monkeypatch):
    # Mock serial.Serial for ArduinoCommunicator
    import serial
    monkeypatch.setattr(serial, "Serial", MockSerialIntegration)

    # Start the main processing threads
    processing_main.start_processing_threads()
    time.sleep(2) # Give threads some time to initialize

    yield # This is where the tests run

    # Teardown (optional, if you need to stop threads explicitly)
    # For this setup, threads are daemons and will exit with the main process.

@pytest.fixture
def client():
    # Create a test client for the Flask app
    flask_app.app.config['TESTING'] = True
    with flask_app.app.test_client() as client:
        yield client

def test_flask_app_index_page(client, setup_integration_environment):
    response = client.get('/')
    assert response.status_code == 200
    # React SPA'da root div kontrolü
    assert b'id="root"' in response.data or b'<!DOCTYPE html>' in response.data

def test_flask_app_status_feed(client, setup_integration_environment):
    response = client.get('/api/status')
    assert response.status_code == 200
    data = response.get_json()
    assert "arduino_status" in data
    assert "detection_results" in data
    assert "lane_results" in data
    assert "obstacle_results" in data
    assert "direction_data" in data

    # Check if some initial data is present (even if placeholder)
    assert data["arduino_status"] in ["Connected", "Disconnected"]

    # Verify that processing threads are populating data (eventually)
    # This might require a short sleep if data propagation is not instantaneous
    # For now, just check for dictionary presence
    assert isinstance(data["detection_results"], dict)
    assert isinstance(data["lane_results"], dict)
    assert isinstance(data["obstacle_results"], dict)
    assert isinstance(data["direction_data"], dict)

# You can add more integration tests here, e.g., for video_feed if you can mock OpenCV frames
