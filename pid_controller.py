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

        # Apply output limits and prevent integral windup
        low, high = self.output_limits
        if low is not None and output < low:
            output = low
            # Prevent integral windup by backing off the integral term
            if self.ki != 0:
                self._integral = (output - self.kp * error - self.kd * derivative) / self.ki
        elif high is not None and output > high:
            output = high
            # Prevent integral windup by backing off the integral term
            if self.ki != 0:
                self._integral = (output - self.kp * error - self.kd * derivative) / self.ki

        return output
