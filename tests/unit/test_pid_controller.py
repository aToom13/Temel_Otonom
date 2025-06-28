import pytest
from core.algorithms.pid_controller import PIDController

def test_pid_basic():
    pid = PIDController(kp=1.0, ki=0.1, kd=0.05, setpoint=10)
    output = pid.compute(measurement=8, dt=1.0)
    assert isinstance(output, float)
    assert output != 0

def test_pid_zero_error():
    pid = PIDController(kp=1.0, ki=0.1, kd=0.05, setpoint=5)
    output = pid.compute(measurement=5, dt=1.0)
    assert output == 0
