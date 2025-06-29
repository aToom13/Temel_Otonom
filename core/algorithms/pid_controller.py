class PIDController:
    def __init__(self, kp, ki, kd, setpoint=0, output_limits=(None, None)):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.setpoint = setpoint
        self._last_error = 0
        self._integral = 0
        self.output_limits = output_limits

    def reset(self):
        self._last_error = 0
        self._integral = 0

    def compute(self, measurement, dt):
        error = self.setpoint - measurement
        self._integral += error * dt
        derivative = (error - self._last_error) / dt if dt > 0 else 0
        output = self.kp * error + self.ki * self._integral + self.kd * derivative
        self._last_error = error
        min_out, max_out = self.output_limits
        if min_out is not None:
            output = max(min_out, output)
        if max_out is not None:
            output = min(max_out, output)
        return output
