import pytest
from modules.arduino_cominicator import ArduinoCommunicator
import serial

# Mock serial.Serial to prevent actual serial port connection during tests
class MockSerial:
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
        if self.in_waiting > 0:
            self.in_waiting -= 1
            return b"ACK\n"
        return b""

    def reset_input_buffer(self):
        pass

    def reset_output_buffer(self):
        pass

@pytest.fixture
def mock_serial(monkeypatch):
    # Replace serial.Serial with our mock class
    monkeypatch.setattr(serial, "Serial", MockSerial)

@pytest.fixture
def arduino_communicator(mock_serial):
    # Initialize ArduinoCommunicator, it will use the mocked serial
    comm = ArduinoCommunicator(port="/dev/ttyUSB_MOCK", baud_rate=9600)
    yield comm
    comm.disconnect()

def test_arduino_communicator_initialization(arduino_communicator):
    assert arduino_communicator is not None
    assert arduino_communicator.is_connected() == True

def test_arduino_communicator_send_data(arduino_communicator):
    test_data = {
        "angle": 45,
        "speed": 100,
        "status": "Düz"
    }
    success = arduino_communicator.send_data(test_data)
    assert success == True
    # Check if the mock serial received the correct data
    expected_message = b"A45S100V0\n"
    assert arduino_communicator.serial_connection.written_data[-1] == expected_message

def test_arduino_communicator_send_data_disconnected(arduino_communicator):
    arduino_communicator.disconnect()
    test_data = {"angle": 0, "speed": 0, "status": "Dur"}
    success = arduino_communicator.send_data(test_data)
    assert success == False

def test_arduino_communicator_read_data(arduino_communicator):
    arduino_communicator.serial_connection.in_waiting = 1 # Simulate data available
    data = arduino_communicator.read_data()
    assert data == "ACK"

