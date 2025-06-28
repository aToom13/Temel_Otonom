from core.algorithms.pid_controller import PIDController

class VehicleController:
    def __init__(self, steering_pid_params, speed_pid_params):
        self.steering_pid = PIDController(**steering_pid_params)
        self.speed_pid = PIDController(**speed_pid_params)

    def compute_control(self, current_steering, current_speed, target_steering, target_speed, dt):
        steering_cmd = self.steering_pid.compute(current_steering - target_steering, dt)
        speed_cmd = self.speed_pid.compute(current_speed - target_speed, dt)
        return steering_cmd, speed_cmd
