import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pybullet as p
import pybullet_data
import time

class BallBalanceEnv(gym.Env):
    """
    SAC-friendly Ball Balancing Environment with improved reward structure
    - Actions: Normalized in [-1, 1] (scaled to delta angles internally)
    - Rewards: Designed to encourage both survival and good balancing
    - Physics: Randomized friction & mass for robustness
    """
    
    def __init__(self, render_mode="human", max_steps=2000, control_freq=60,
                 add_obs_noise=False, obs_noise_std_pos=0.001, obs_noise_std_vel=0.02,
                 friction_low=0.18, friction_high=0.26,
                 ball_mass_low=0.0018, ball_mass_high=0.0022,
                 use_pid_guidance=False, pid_kp=1.35, pid_kd=0.18, pid_guidance_weight=0.2):
        super().__init__()
        
        self.render_mode = render_mode
        self.max_steps = max_steps
        self.current_step = 0

        # Table limits
        self.limit_angle = np.radians(9)        # Max tilt ±9°
        self.max_delta_angle = self.limit_angle / 2  # Max change per step

        # SAC-friendly action space: normalized [-1, 1]
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )
        
        # Observation space: [ball_x, ball_y, ball_vx, ball_vy, table_pitch, table_roll]
        self.table_radius = 0.125
        self.observation_space = spaces.Box(
            low=np.array([-0.15, -0.15, -2.0, -2.0, -0.2, -0.2]),
            high=np.array([0.15, 0.15, 2.0, 2.0, 0.2, 0.2]),
            dtype=np.float32
        )
        
        # Timing
        self.physics_freq = 240
        self.control_freq = control_freq
        self.physics_dt = 1.0 / self.physics_freq
        self.control_dt = 1.0 / self.control_freq
        self.physics_steps_per_control = self.physics_freq // self.control_freq
        
        # Ball & table properties
        self.ball_radius = 0.0075
        self.table_size = 0.125
        self.spawn_range = 0.10  # default spawn range (radius on table)
        
        # State
        self.table_pitch = 0.0
        self.table_roll = 0.0
        self.prev_ball_pos = None
        self.prev_actions = []
        self.small_action_streak = 0
        self.near_center_streak = 0
        
        # Reward tracking for normalization
        self.episode_rewards = []
        
        # PyBullet
        self.physics_client = None
        self.table_id = None
        self.ball_id = None
        self.plane_id = None
        self.base_id = None

        # Observation noise (for sim-to-real robustness)
        self.add_obs_noise = add_obs_noise
        self.obs_noise_std_pos = float(obs_noise_std_pos)
        self.obs_noise_std_vel = float(obs_noise_std_vel)

        # Domain randomization ranges
        self.friction_low = float(friction_low)
        self.friction_high = float(friction_high)
        self.ball_mass_low = float(ball_mass_low)
        self.ball_mass_high = float(ball_mass_high)

        # PID guidance (for reward shaping / imitation of baseline controller)
        self.use_pid_guidance = bool(use_pid_guidance)
        self.pid_kp = float(pid_kp)
        self.pid_kd = float(pid_kd)
        self.pid_guidance_weight = float(pid_guidance_weight)
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.current_step = 0
        self.table_pitch = 0.0
        self.table_roll = 0.0
        self.prev_ball_pos = None
        self.prev_actions = []
        self.small_action_streak = 0
        self.near_center_streak = 0
        self.episode_rewards = []

        # Disconnect previous physics
        if self.physics_client is not None:
            p.disconnect(self.physics_client)
        
        # Connect to PyBullet
        if self.render_mode == "human":
            self.physics_client = p.connect(p.GUI)
            p.resetDebugVisualizerCamera(
                cameraDistance=0.6,
                cameraYaw=30,
                cameraPitch=-45,
                cameraTargetPosition=[0, 0, 0.06]
            )
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        else:
            self.physics_client = p.connect(p.DIRECT)
        
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        
        # Ground plane
        ground_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[2, 2, 0.01])
        ground_visual = p.createVisualShape(p.GEOM_BOX, halfExtents=[2, 2, 0.01],
                                            rgbaColor=[0.4, 0.4, 0.4, 1])
        self.plane_id = p.createMultiBody(0, ground_shape, ground_visual, [0, 0, -0.01])
        
        # Base
        self.base_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.05, 0.05, 0.03]),
            baseVisualShapeIndex=p.createVisualShape(p.GEOM_BOX, halfExtents=[0.05, 0.05, 0.03],
                                                     rgbaColor=[0.3, 0.3, 0.3, 1]),
            basePosition=[0, 0, 0.02],
        )
        
        # Table
        table_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.125, 0.125, 0.004])
        table_visual = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.125, 0.125, 0.004],
                                           rgbaColor=[0.1, 0.1, 0.1, 1])
        self.table_start_pos = [0, 0, 0.06]
        self.table_id = p.createMultiBody(1.0, table_shape, table_visual, self.table_start_pos)
        
        # Ball spawn
        spawn_range = self.spawn_range  # Use configurable spawn range
        ball_start_pos = [
            np.random.uniform(-spawn_range, spawn_range),
            np.random.uniform(-spawn_range, spawn_range),
            0.5
        ]
        
        # Ball
        self.ball_id = p.createMultiBody(
            baseMass=0.002,
            baseCollisionShapeIndex=p.createCollisionShape(p.GEOM_SPHERE, radius=self.ball_radius),
            baseVisualShapeIndex=p.createVisualShape(p.GEOM_SPHERE, radius=self.ball_radius,
                                                     rgbaColor=[0.9, 0.9, 0.9, 1]),
            basePosition=ball_start_pos
        )

        # Randomize physics params
        ball_mass = np.random.uniform(self.ball_mass_low, self.ball_mass_high)
        ball_friction = np.random.uniform(self.friction_low, self.friction_high)
        table_friction = np.random.uniform(self.friction_low, self.friction_high)
        p.changeDynamics(self.ball_id, -1, lateralFriction=ball_friction, rollingFriction=0.05,
                         spinningFriction=0.05, linearDamping=0.1, angularDamping=0.08,
                         restitution=0.3, mass=ball_mass, contactStiffness=2500,
                         contactDamping=80)
        p.changeDynamics(self.table_id, -1, lateralFriction=table_friction, rollingFriction=0.05,
                         restitution=0.3)
        
        axis_length = 0.1
        p.addUserDebugLine([0, 0, 0.065], [axis_length, 0, 0.065], [1, 0, 0], lineWidth=3)  # X-axis red
        p.addUserDebugLine([0, 0, 0.065], [0, axis_length, 0.065], [0, 1, 0], lineWidth=3)  # Y-axis green  
        p.addUserDebugLine([0, 0, 0.065], [0, 0, 0.065 + axis_length], [0, 0, 1], lineWidth=3)  # Z-axis blue
        p.addUserDebugText("X", [axis_length, 0, 0.065], textColorRGB=[1, 0, 0], textSize=2)
        p.addUserDebugText("Y", [0, axis_length, 0.065], textColorRGB=[0, 1, 0], textSize=2)
        p.addUserDebugText("Z", [0, 0, 0.065 + axis_length], textColorRGB=[0, 0, 1], textSize=2)

        # Let ball settle
        for _ in range(60):
            p.stepSimulation()
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        self.current_step += 1

        # Scale normalized [-1,1] action to delta radians
        delta_pitch = action[0] * self.max_delta_angle
        delta_roll = action[1] * self.max_delta_angle

        self.table_pitch = np.clip(self.table_pitch + delta_pitch, -self.limit_angle, self.limit_angle)
        self.table_roll = np.clip(self.table_roll + delta_roll, -self.limit_angle, self.limit_angle)

        quat = p.getQuaternionFromEuler([self.table_pitch, self.table_roll, 0])
        p.resetBasePositionAndOrientation(self.table_id, self.table_start_pos, quat)

        for _ in range(self.physics_steps_per_control):
            p.stepSimulation()
            if self.render_mode == "human":
                time.sleep(self.physics_dt)

        obs = self._get_observation()
        reward = self._calculate_reward_improved(obs, action)
        terminated = self._is_terminated(obs)
        truncated = self.current_step >= self.max_steps
        info = self._get_info()
        
        self.episode_rewards.append(reward)
        
        return obs, reward, terminated, truncated, info

    def _get_observation(self):
        ball_pos, _ = p.getBasePositionAndOrientation(self.ball_id)
        ball_x, ball_y, _ = ball_pos
        ball_vx, ball_vy = 0.0, 0.0
        if self.prev_ball_pos is not None:
            ball_vx = (ball_x - self.prev_ball_pos[0]) / self.control_dt
            ball_vy = (ball_y - self.prev_ball_pos[1]) / self.control_dt
        self.prev_ball_pos = [ball_x, ball_y]

        # Optional observation noise (positions/velocities only)
        if self.add_obs_noise:
            noise_px = np.random.normal(0.0, self.obs_noise_std_pos)
            noise_py = np.random.normal(0.0, self.obs_noise_std_pos)
            noise_vx = np.random.normal(0.0, self.obs_noise_std_vel)
            noise_vy = np.random.normal(0.0, self.obs_noise_std_vel)
            ball_x_noisy = ball_x + noise_px
            ball_y_noisy = ball_y + noise_py
            ball_vx_noisy = ball_vx + noise_vx
            ball_vy_noisy = ball_vy + noise_vy
        else:
            ball_x_noisy, ball_y_noisy = ball_x, ball_y
            ball_vx_noisy, ball_vy_noisy = ball_vx, ball_vy

        return np.array([ball_x_noisy, ball_y_noisy, ball_vx_noisy, ball_vy_noisy, self.table_pitch, self.table_roll], dtype=np.float32)

    def _calculate_reward_improved(self, obs, action):
        """
        Improved reward function that encourages both survival and good balancing
        """
        ball_x, ball_y, ball_vx, ball_vy, table_pitch, table_roll = obs
        dist = np.sqrt(ball_x**2 + ball_y**2)
        
        # Terminal penalty for falling off
        if dist > self.table_radius:
            return -100.0  # Large negative, but not overwhelming for episode return
        
        # Base survival reward (positive for staying on table)
        survival_reward = 1.0
        
        # Distance penalty (scaled to be less dominant)
        # Use exponential decay for smoother gradient
        distance_penalty = -2.0 * (dist / self.table_radius)**2
        
        # Velocity penalty (encourage stability)
        velocity = np.sqrt(ball_vx**2 + ball_vy**2)
        velocity_penalty = -0.5 * min(velocity, 2.0)  # Cap to prevent explosion
        
        # Control effort penalty (encourage smooth control)
        action_magnitude = np.linalg.norm(action)
        control_penalty = -0.05 * action_magnitude
        
        # Angle penalty (penalize extreme tilts)
        angle_magnitude = np.sqrt(table_pitch**2 + table_roll**2)
        angle_penalty = -0.3 * (angle_magnitude / self.limit_angle)
        
        # Bonuses for excellent performance
        center_bonus = 0.0
        if dist < 0.01:  # Very close to center
            center_bonus = 3.0
        elif dist < 0.03:  # Near center
            center_bonus = 0.5
        
        # Stability bonus (low velocity near center)
        stability_bonus = 0.0
        if dist < 0.05 and velocity < 0.1:
            stability_bonus = 1.5
        
        # PID guidance (encourage table angles to be close to a PD baseline)
        guidance_bonus = 0.0
        if self.use_pid_guidance:
            # PD baseline angles (absolute), clipped to physical limits
            # Sign convention matches compare_control PID: pitch ~ -PID(x), roll ~ +PID(y)
            pid_pitch_target = - (self.pid_kp * ball_x + self.pid_kd * ball_vx)
            pid_roll_target  =   (self.pid_kp * ball_y + self.pid_kd * ball_vy)
            pid_pitch_target = float(np.clip(pid_pitch_target, -self.limit_angle, self.limit_angle))
            pid_roll_target  = float(np.clip(pid_roll_target,  -self.limit_angle, self.limit_angle))

            angle_error = np.sqrt((table_pitch - pid_pitch_target)**2 + (table_roll - pid_roll_target)**2)
            # Negative penalty on error (scaled); use exp to soften far-away gradients
            guidance_bonus = - self.pid_guidance_weight * angle_error

        # Combine all components
        total_reward = (
            survival_reward +
            distance_penalty +
            velocity_penalty +
            control_penalty +
            angle_penalty +
            center_bonus +
            stability_bonus +
            guidance_bonus
        )
        
        return float(np.clip(total_reward, -100, 8))
    
    def _calculate_reward_sparse(self, obs, action):
        """
        Alternative: Sparse reward function (simpler for some algorithms)
        """
        ball_x, ball_y, _, _, _, _ = obs
        dist = np.sqrt(ball_x**2 + ball_y**2)
        
        # Terminal penalty
        if dist > self.table_radius:
            return -1.0
        
        # Simple distance-based reward
        if dist < 0.02:
            return 1.0
        elif dist < 0.05:
            return 0.1
        else:
            return 0.0

    def _is_terminated(self, obs):
        ball_x, ball_y, _, _, _, _ = obs
        dist = np.sqrt(ball_x**2 + ball_y**2)
        if dist > self.table_radius:
            return True
        ball_pos, _ = p.getBasePositionAndOrientation(self.ball_id)
        return ball_pos[2] < 0.05

    def _get_info(self):
        ball_pos, _ = p.getBasePositionAndOrientation(self.ball_id)
        dist = np.sqrt(ball_pos[0]**2 + ball_pos[1]**2)
        
        # Add episode return to info for monitoring
        episode_return = sum(self.episode_rewards) if self.episode_rewards else 0
        
        return {
            "distance_from_center": dist,
            "ball_position": ball_pos,
            "table_angles": [self.table_pitch, self.table_roll],
            "step": self.current_step,
            "episode_return": episode_return
        }

    def close(self):
        if self.physics_client is not None:
            p.disconnect(self.physics_client)
            self.physics_client = None

    # ----- Curriculum support -----
    def set_curriculum_stage(self, stage: str):
        """Adjust environment difficulty. Stages: 'easy', 'medium', 'hard'"""
        stage = stage.lower()
        if stage == 'easy':
            self.spawn_range = 0.05
            self.max_delta_angle = self.limit_angle / 4  # smaller per-step change
        elif stage == 'medium':
            self.spawn_range = 0.08
            self.max_delta_angle = self.limit_angle / 3
        else:  # 'hard' or default
            self.spawn_range = 0.10
            self.max_delta_angle = self.limit_angle / 2
