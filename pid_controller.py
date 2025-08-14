class PIDController:
    def __init__(self, kp, ki, kd, setpoint=0, output_limits=(None, None)):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.setpoint = setpoint
        self._integral = 0
        self._prev_error = None
        self._prev_measurement = None
        self.output_limits = output_limits

    def reset(self):
        self._integral = 0
        self._prev_error = None
        self._prev_measurement = None

    def update(self, measurement, dt):
        error = self.setpoint - measurement

        # Derivative on measurement to avoid derivative kick
        derivative = 0 if self._prev_measurement is None else -(measurement - self._prev_measurement) / dt
        self._prev_measurement = measurement

        # Only integrate if not saturated or driving output back toward unsaturation
        output_unsat = self.kp * error + self.ki * self._integral + self.kd * derivative
        low, high = self.output_limits
        if (low is None or output_unsat > low) and (high is None or output_unsat < high):
            self._integral += error * dt

        # Recalculate with updated integral
        output = self.kp * error + self.ki * self._integral + self.kd * derivative

        # Clamp output
        if low is not None:
            output = max(low, output)
        if high is not None:
            output = min(high, output)

        return output