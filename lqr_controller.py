import numpy as np
from scipy.linalg import solve_continuous_are

class LQRController:
    def __init__(self,
                 velocity_smoothing: bool = True,
                 Q: np.ndarray | None = None,
                 R: np.ndarray | None = None,
                 use_gravity_model: bool = True,
                 servo_limit_deg: float = 9.0,
                 smoothing_alpha: float = 0.8):
        """
        Linear-Quadratic Regulator for the ball-table system.
        - State: [ball_x, ball_vx, ball_y, ball_vy]
        - Input: [table_pitch, table_roll] (radians)
        """
        self.g = 9.81 if use_gravity_model else 1.0

        # Continuous-time linearized model around small angles
        self.A = np.array([
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 0.0]
        ])

        # For small angles: ball_acc ≈ g * table_angle
        self.B = np.array([
            [0.0, 0.0],
            [self.g, 0.0],
            [0.0, 0.0],
            [0.0, self.g]
        ])

        # Tuned cost matrices (good default for your hardware)
        self.Q = Q if Q is not None else np.diag([25, 1.0, 25, 1])
        self.R = R if R is not None else np.diag([20, 20])

        

        # Servo output limits (match table limits)
        self.servo_limit = np.radians(servo_limit_deg)

        # Velocity smoothing (exponential moving average)
        self.velocity_smoothing = velocity_smoothing
        self.smoothing_alpha = float(np.clip(smoothing_alpha, 0.0, 1.0))  # previous weight
        self.prev_ball_vx = None
        self.prev_ball_vy = None

        # Compute gain matrix K
        self.K = self.compute_lqr_gain(self.A, self.B, self.Q, self.R)

    def compute_lqr_gain(self, A, B, Q, R):
        """Solve the continuous-time Algebraic Riccati Equation and compute LQR gain"""
        P = solve_continuous_are(A, B, Q, R)
        K = np.linalg.inv(R) @ B.T @ P
        return K

    def control(self, ball_x, ball_y, ball_vx, ball_vy,
                setpoint_x: float = 0.0, setpoint_y: float = 0.0,
                setpoint_vx: float = 0.0, setpoint_vy: float = 0.0):
        """Compute LQR control output given current state and setpoint."""
        # Optional velocity smoothing
        if self.velocity_smoothing:
            if self.prev_ball_vx is not None:
                a = self.smoothing_alpha  # weight on previous value
                ball_vx = a * self.prev_ball_vx + (1.0 - a) * ball_vx
                ball_vy = a * self.prev_ball_vy + (1.0 - a) * ball_vy
            self.prev_ball_vx = ball_vx
            self.prev_ball_vy = ball_vy

        x = np.array([ball_x, ball_vx, ball_y, ball_vy], dtype=float)
        x_desired = np.array([setpoint_x, setpoint_vx, setpoint_y, setpoint_vy], dtype=float)
        u = -self.K @ (x - x_desired)

        # Clip output to servo limits
        pitch_angle = float(np.clip(u[0], -self.servo_limit, self.servo_limit))
        roll_angle = float(np.clip(u[1], -self.servo_limit, self.servo_limit))

        return pitch_angle, roll_angle