import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pybullet as p
import pybullet_data
import time
from typing import Tuple, Dict, Any


class BallBalanceEnv(gym.Env):
    """
    Reinforcement Learning Environment for Ball Balancing Table
    
    Observation Space:
    - Ball position (x, y) - from camera/sensor
    - Ball velocity (vx, vy) - estimated from position differences (realistic)
    - Table angles (pitch, roll)
    
    Action Space:
    - Table pitch angle change
    - Table roll angle change
    """
    
    def __init__(self, render_mode="human", max_steps=2000, control_freq=50):
        super().__init__()
        
        self.render_mode = render_mode
        self.max_steps = max_steps
        self.current_step = 0

        
        # Action space: pitch and roll angle changes (in radians) - smaller for smoother control
        self.action_space = spaces.Box(
            low=-0.05, high=0.05, shape=(2,), dtype=np.float32  
        )
        
        # Observation space: [ball_x, ball_y, ball_vx, ball_vy, table_pitch, table_roll] - ESTIMATED VELOCITY
        # Updated bounds for 25cm table
        self.observation_space = spaces.Box(
            low=np.array([-0.15, -0.15, -2.0, -2.0, -0.1, -0.1]),
            high=np.array([0.15, 0.15, 2.0, 2.0, 0.1, 0.1]),
            dtype=np.float32
        )
        
        # Timing parameters
        self.physics_freq = 240  # Hz - physics simulation frequency
        self.control_freq = control_freq  # Hz - control update frequency (default 50 Hz like servos)
        self.physics_dt = 1.0 / self.physics_freq  # Physics timestep
        self.control_dt = 1.0 / self.control_freq  # Control timestep
        self.physics_steps_per_control = self.physics_freq // self.control_freq  # Steps per control update
        
        # Physics parameters
        self.ball_radius = 0.0075 # 7.5mm radius 
        self.table_size = 0.125  # 25cm table (radius from center to edge)
        
        # State tracking
        self.table_pitch = 0.0
        self.table_roll = 0.0
        self.prev_ball_pos = None  # For velocity estimation
        self.prev_observation_time = None  # For proper velocity estimation timing
        self.prev_table_pitch = 0.0
        self.prev_table_roll = 0.0
        self.prev_actions = []  # Track action history for oscillation detection
        self.prev_action = None  # Track previous action for jerk penalty
        
        # Initialize PyBullet
        self.physics_client = None
        self.table_id = None
        self.ball_id = None
        self.plane_id = None
        self.base_id = None
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.current_step = 0
        self.table_pitch = 0.0
        self.table_roll = 0.0
        self.prev_ball_pos = None  # Reset for velocity estimation
        self.prev_observation_time = None  # Reset timing for velocity estimation
        self.prev_table_pitch = 0.0
        self.prev_table_roll = 0.0
        self.prev_actions = []  # Reset action history
        self.prev_action = None  # Reset previous action for jerk penalty

        
        # Disconnect existing physics client if it exists
        if self.physics_client is not None:
            p.disconnect(self.physics_client)
        
        # Initialize PyBullet
        if self.render_mode == "human":
            self.physics_client = p.connect(p.GUI)
            
            # Clean, professional camera setup
            p.resetDebugVisualizerCamera(
                cameraDistance=0.6,        # Closer view for better detail
                cameraYaw=30,              # Slight angle for better perspective
                cameraPitch=-45,           # Looking down at optimal angle
                cameraTargetPosition=[0, 0, 0.06]  # Focus on table center
            )
            
            # Remove GUI clutter for clean appearance
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)                    # Hide control panel
            p.configureDebugVisualizer(p.COV_ENABLE_TINY_RENDERER, 0)          # Better rendering quality
            p.configureDebugVisualizer(p.COV_ENABLE_MOUSE_PICKING, 0)          # Disable mouse interaction
            p.configureDebugVisualizer(p.COV_ENABLE_KEYBOARD_SHORTCUTS, 0)     # Disable keyboard shortcuts
            p.configureDebugVisualizer(p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 0)  # Clean view
            p.configureDebugVisualizer(p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 0)   # Clean view
            p.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW, 0)     # Clean view
        else:
            self.physics_client = p.connect(p.DIRECT)
            
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        
        # Create custom gray ground plane instead of default blue/white checkerboard
        ground_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[2, 2, 0.01])
        ground_visual = p.createVisualShape(p.GEOM_BOX, halfExtents=[2, 2, 0.01], 
                                          rgbaColor=[0.4, 0.4, 0.4, 1])  # Clean gray
        self.plane_id = p.createMultiBody(baseMass=0, 
                                        baseCollisionShapeIndex=ground_shape,
                                        baseVisualShapeIndex=ground_visual,
                                        basePosition=[0, 0, -0.01])
        
        # Base support - darker gray for better contrast
        self.base_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.05, 0.05, 0.03]),
            baseVisualShapeIndex=p.createVisualShape(p.GEOM_BOX, halfExtents=[0.05, 0.05, 0.03], rgbaColor=[0.3, 0.3, 0.3, 1]),
            basePosition=[0, 0, 0.02],
        )
        
        # Table - 25cm x 25cm (halfExtents = total_size / 2) - sleek dark surface
        table_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.125, 0.125, 0.004])
        table_visual = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.125, 0.125, 0.004], 
                                         rgbaColor=[0.1, 0.1, 0.1, 1],     # Dark surface
                                         specularColor=[0.2, 0.2, 0.2])    # Slight metallic look
        self.table_start_pos = [0, 0, 0.06]
        self.table_id = p.createMultiBody(1.0, table_shape, table_visual, self.table_start_pos)
        
    
        # Ball - randomize initial position slightly
        if options and 'ball_start_pos' in options:
            ball_start_pos = options['ball_start_pos']
        else:
            # Random start position within reasonable bounds - 25cm table
            ball_start_pos = [
                np.random.uniform(-0.12, 0.12),  # Stay within 24cm range for safety
                np.random.uniform(-0.12, 0.12),
                0.5
            ]

    
        axis_length = 0.1
        p.addUserDebugLine([0, 0, 0.065], [axis_length, 0, 0.065], [1, 0, 0], lineWidth=3)  # X-axis red
        p.addUserDebugLine([0, 0, 0.065], [0, axis_length, 0.065], [0, 1, 0], lineWidth=3)  # Y-axis green  
        p.addUserDebugLine([0, 0, 0.065], [0, 0, 0.065 + axis_length], [0, 0, 1], lineWidth=3)  # Z-axis blue
        p.addUserDebugText("X", [axis_length, 0, 0.065], textColorRGB=[1, 0, 0], textSize=2)
        p.addUserDebugText("Y", [0, axis_length, 0.065], textColorRGB=[0, 1, 0], textSize=2)
        p.addUserDebugText("Z", [0, 0, 0.065 + axis_length], textColorRGB=[0, 0, 1], textSize=2)
        
        self.ball_id = p.createMultiBody(
            baseMass=0.002,  # 
            baseCollisionShapeIndex=p.createCollisionShape(p.GEOM_SPHERE, radius=self.ball_radius),
            baseVisualShapeIndex=p.createVisualShape(
                p.GEOM_SPHERE,
                radius=self.ball_radius,
                rgbaColor=[0.9, 0.9, 0.9, 1],
                specularColor=[0.8, 0.8, 0.8]
            ),
            basePosition=ball_start_pos
        )

        # 🎯 CRITICAL: Apply realistic physics parameters for PLEXIGLASS table
        # Plexiglass is much more slippery than wood/metal surfaces
        p.changeDynamics(
            self.ball_id, -1,
            lateralFriction=0.22,         # Slightly higher for more ground resistance
            rollingFriction=0.05,        # Increase this to resist rolling motion
            spinningFriction=0.05,      # Increase to resist spin
            linearDamping=0.1,           # Simulates air resistance / velocity damping
            angularDamping=0.08,         # Slows down spinning over time
            restitution=0.3,             # Lower if you want less bounce
            contactStiffness=2500,       # Already OK for plexiglass
            contactDamping=80            # Increase to dissipate energy on contact
        )
        
        # Plexiglass table surface properties
        p.changeDynamics(
            self.table_id, -1,
            lateralFriction=0.22,       # Match the ball
            rollingFriction=0.05,
            restitution=0.3            # Match ball bounce
        )

        
        # Let the ball settle
        for _ in range(100):
            p.stepSimulation()
        
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, info
    
    def step(self, action):
        self.current_step += 1
        
        # Store previous angles for smooth movement penalty
        self.prev_table_pitch = self.table_pitch
        self.prev_table_roll = self.table_roll
        
        # Track action history for oscillation detection (keep last 6 actions)
        self.prev_actions.append(action.copy())
        if len(self.prev_actions) > 6:
            self.prev_actions.pop(0)
        
        # Apply action (change in table angles)
        self.table_pitch += action[0]
        self.table_roll += action[1]
        
        # Clip angles to reasonable limits
        self.table_pitch = np.clip(self.table_pitch, -0.1920, 0.1920)
        self.table_roll = np.clip(self.table_roll, -0.1920, 0.1920)

        # Update table orientation with ACTUAL servo positions 
        quat = p.getQuaternionFromEuler([self.table_pitch, self.table_roll, 0])
        p.resetBasePositionAndOrientation(self.table_id, self.table_start_pos, quat)
        
        # Step simulation multiple times to maintain proper physics frequency
        for _ in range(self.physics_steps_per_control):
            p.stepSimulation()
            if self.render_mode == "human":
                time.sleep(self.physics_dt)
        
        # Get new observation
        observation = self._get_observation()
        
        # Calculate reward
        reward = self._calculate_reward(observation, action)
        
        # Check termination conditions
        terminated = self._is_terminated(observation)
        truncated = self.current_step >= self.max_steps
        
        info = self._get_info()
        
        return observation, reward, terminated, truncated, info
    
    def _get_observation(self):
        # Ball position from sensor/camera
        ball_pos, _ = p.getBasePositionAndOrientation(self.ball_id)
        ball_x, ball_y, ball_z = ball_pos
        
        # Estimate velocity from position differences using actual control timestep
        ball_vx, ball_vy = 0.0, 0.0
        if self.prev_ball_pos is not None:
            # Use control timestep for proper velocity estimation
            ball_vx = (ball_x - self.prev_ball_pos[0]) / self.control_dt
            ball_vy = (ball_y - self.prev_ball_pos[1]) / self.control_dt
        
        # Update previous position for next velocity calculation
        self.prev_ball_pos = [ball_x, ball_y]
        
        # Return position + estimated velocity + table angles
        observation = np.array([
            ball_x, ball_y, ball_vx, ball_vy, self.table_pitch, self.table_roll
        ], dtype=np.float32)
        
        return observation
    
    def _calculate_reward(self, observation, action):
        ball_x, ball_y, ball_vx, ball_vy, table_pitch, table_roll = observation
        distance_from_center = np.sqrt(ball_x**2 + ball_y**2)

        if distance_from_center > self.table_size:
            return -10.0  # Fail case

        # === 1. Position reward (normalized: max ~ +4.0) ===
        position_reward = max(0.0, 4.0 * (1.0 - (distance_from_center / self.table_size)))

        # === 2. Time bonus (small constant) ===
        time_bonus = 0.1  # Increased from 0.05 to enhance survival incentive

        # === 3. Action penalty (softer, quadratic) ===
        action_magnitude = np.linalg.norm(action)
        action_penalty = -10.0 * action_magnitude**2  # Down from -20

        # Soft limit penalty
        if np.any(np.abs(action) > 0.045):
            action_penalty += -1.0  # Reduced from -2

        # === 4. Bang-bang streak penalty (smoothed) ===
        if not hasattr(self, 'max_action_streak'):
            self.max_action_streak = 0
        if np.any(np.abs(action) > 0.048):
            self.max_action_streak += 1
        else:
            self.max_action_streak = 0

        if self.max_action_streak >= 3:
            action_penalty += -2.0 - 1.0 * (self.max_action_streak - 3)  # Smoothed penalty instead of cliff

        # === 5. Oscillation / circular motion penalty ===
        oscillation_penalty = 0.0
        if len(self.prev_actions) >= 6:
            recent_actions = self.prev_actions[-6:] + [action]
            pitch_signs = [np.sign(a[0]) for a in recent_actions if abs(a[0]) > 0.03]
            roll_signs = [np.sign(a[1]) for a in recent_actions if abs(a[1]) > 0.03]

            def is_alternating(signs):
                if len(signs) < 4:
                    return False
                return sum(signs[i] != signs[i-1] for i in range(1, len(signs))) >= 3

            if is_alternating(pitch_signs) or is_alternating(roll_signs):
                oscillation_penalty += -1.0  # Reduced from -2

            mags = [np.linalg.norm(a) for a in recent_actions]
            if np.mean(mags) > 0.03:
                angles = [np.arctan2(a[1], a[0]) for a in recent_actions if np.linalg.norm(a) > 0.01]
                if len(angles) >= 5:
                    diffs = [abs(angles[i] - angles[i-1]) for i in range(1, len(angles))]
                    avg_diff = np.mean(diffs)
                    if 0.3 < avg_diff < 2.0:
                        oscillation_penalty += -1.5  # Reduced from -3

        # === 6. Smoothness bonus (kept as-is) ===
        if not hasattr(self, 'small_action_streak'):
            self.small_action_streak = 0
        if action_magnitude < 0.012:
            self.small_action_streak += 1
        else:
            self.small_action_streak = 0

        smoothness_bonus = 0.0
        if self.small_action_streak >= 8:
            smoothness_bonus = 2.0  # Slightly reduced from 3.0

        # === 7. Prolonged table angle penalty (softened) ===
        table_angle = np.sqrt(table_pitch**2 + table_roll**2)
        if not hasattr(self, 'table_angle_history'):
            self.table_angle_history = []
        self.table_angle_history.append(table_angle)
        if len(self.table_angle_history) > 8:
            self.table_angle_history.pop(0)

        avg_table_angle = np.mean(self.table_angle_history)
        prolonged_angle_penalty = -2.0 * avg_table_angle  # Reduced from -3.0

        # === 8. Optimality penalty ===
        optimal_action = self._calculate_optimal_action(ball_x, ball_y, ball_vx, ball_vy)
        optimality_penalty = -5.0 * np.linalg.norm(action - optimal_action)  # Weight controls impact

        # === Final Reward ===
        total_reward = (
            position_reward +
            time_bonus +
            action_penalty +
            oscillation_penalty +
            smoothness_bonus +
            prolonged_angle_penalty +
            optimality_penalty  
        )

        # === Normalize to [-10, 10] ===
        total_reward = np.clip(total_reward, -10.0, 10.0)

        # Update prev action
        self.prev_action = action.copy()

        return total_reward

    
    def _calculate_optimal_action(self, ball_x, ball_y, ball_vx, ball_vy):
        """Calculate the theoretically optimal action for current ball state"""
        # Simple PD controller logic: proportional to position + derivative (velocity)
        
        # Proportional gains (how much to tilt based on position error)
        kp = 1.35  # Position gain
        kd = 0.18  # Velocity (derivative) gain
        
        # Calculate desired table tilts to center the ball
        # Tilt table opposite to ball position to "roll" ball toward center
        desired_pitch = -ball_x * kp - ball_vx * kd  # Negative because we want to oppose ball movement
        desired_roll = -ball_y * kp - ball_vy * kd
        
        # Clip to action space limits
        desired_pitch = np.clip(desired_pitch, -0.1920, 0.1920)
        desired_roll = np.clip(desired_roll, -0.1920, 0.1920)
        
        return np.array([desired_pitch, desired_roll], dtype=np.float32)
    
    def _is_terminated(self, observation):
        ball_x, ball_y, ball_vx, ball_vy, table_pitch, table_roll = observation  # Updated for 6D format
        
        # Check if ball fell off the table
        distance_from_center = np.sqrt(ball_x**2 + ball_y**2)
        if distance_from_center > self.table_size:
            return True
        
        # Check if ball fell below table
        ball_pos, _ = p.getBasePositionAndOrientation(self.ball_id)
        if ball_pos[2] < 0.05:  # Below table level
            return True
            
        return False
    
    def _get_info(self):
        ball_pos, _ = p.getBasePositionAndOrientation(self.ball_id)
        distance_from_center = np.sqrt(ball_pos[0]**2 + ball_pos[1]**2)
        
        return {
            "distance_from_center": distance_from_center,
            "ball_position": ball_pos,
            "table_angles": [self.table_pitch, self.table_roll],
            "step": self.current_step
        }
    
    def close(self):
        if self.physics_client is not None:
            p.disconnect(self.physics_client)
            self.physics_client = None
