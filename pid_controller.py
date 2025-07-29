# pid_controller.py
class PIDController:
    def __init__(self, kp, ki, kd, setpoint=0, output_limits=(None, None)):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.setpoint = setpoint
        self._integral = 0
        self._prev_error = None
        self.output_limits = output_limits

    def reset(self):
        self._integral = 0
        self._prev_error = None

    def update(self, measurement, dt):
        error = self.setpoint - measurement
        self._integral += error * dt
        derivative = 0 if self._prev_error is None else (error - self._prev_error) / dt
        self._prev_error = error

        output = self.kp * error + self.ki * self._integral + self.kd * derivative

        low, high = self.output_limits
        if low is not None:
            output = max(low, output)
        if high is not None:
            output = min(high, output)

        return output
