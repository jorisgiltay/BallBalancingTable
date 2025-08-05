import numpy as np
from scipy.linalg import solve_continuous_are

class LQRController:
    def __init__(self, velocity_smoothing=False):
        # Simplified linear model: [ball_x, ball_vx, ball_y, ball_vy]
        self.A = np.array([
            [0, 1, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 0, 0]
        ])

        # Control inputs affect ball acceleration in x and y
        self.B = np.array([
            [0, 0],
            [1, 0],
            [0, 0],
            [0, 1]
        ])

        # State cost matrix — prioritize ball position much more than velocity
        self.Q = np.diag([50, 2, 50, 2])

        # Control effort cost — higher for smoother control
        self.R = np.diag([5, 5])

        # Servo output limits (±3.2°)
        self.servo_limit = 0.0559

        # Optional velocity smoothing
        self.velocity_smoothing = velocity_smoothing
        self.prev_ball_vx = None
        self.prev_ball_vy = None

        # Compute gain matrix K
        self.K = self.compute_lqr_gain(self.A, self.B, self.Q, self.R)

    def compute_lqr_gain(self, A, B, Q, R):
        """Solve the continuous-time Algebraic Riccati Equation and compute LQR gain"""
        P = solve_continuous_are(A, B, Q, R)
        K = np.linalg.inv(R) @ B.T @ P
        return K

    def control(self, ball_x, ball_y, ball_vx, ball_vy):
        """Compute LQR control output given current state, with output clipping and optional velocity smoothing"""
        # Optionally smooth velocity estimates
        if self.velocity_smoothing:
            if self.prev_ball_vx is not None:
                ball_vx = 0.7 * self.prev_ball_vx + 0.3 * ball_vx
                ball_vy = 0.7 * self.prev_ball_vy + 0.3 * ball_vy
            self.prev_ball_vx = ball_vx
            self.prev_ball_vy = ball_vy

        x = np.array([ball_x, ball_vx, ball_y, ball_vy])
        x_desired = np.zeros(4)
        u = -self.K @ (x - x_desired)

        # Clip output to servo limits
        pitch_angle = np.clip(u[0], -self.servo_limit, self.servo_limit)
        roll_angle = np.clip(u[1], -self.servo_limit, self.servo_limit)

        return pitch_angle, roll_angle