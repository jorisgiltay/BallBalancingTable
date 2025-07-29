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
        self.ball_radius = 0.02
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
            # Set up a better camera view
            p.resetDebugVisualizerCamera(
                cameraDistance=0.8,        # Distance from target
                cameraYaw=45,              # Horizontal angle (degrees)
                cameraPitch=-30,           # Vertical angle (degrees, negative = looking down)
                cameraTargetPosition=[0, 0, 0.06]  # Look at the table center
            )
            # Optional: disable some GUI elements for cleaner view
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1)  # Keep GUI
            p.configureDebugVisualizer(p.COV_ENABLE_TINY_RENDERER, 0)  # Better rendering
        else:
            self.physics_client = p.connect(p.DIRECT)
            
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        
        # Create environment objects
        self.plane_id = p.loadURDF("plane.urdf")
        
        # Base support
        self.base_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.05, 0.05, 0.03]),
            baseVisualShapeIndex=p.createVisualShape(p.GEOM_BOX, halfExtents=[0.05, 0.05, 0.03], rgbaColor=[0.5, 0.5, 0.5, 1]),
            basePosition=[0, 0, 0.02],
        )
        
        # Table
        table_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[self.table_size, self.table_size, 0.004])
        table_visual = p.createVisualShape(p.GEOM_BOX, halfExtents=[self.table_size, self.table_size, 0.004], rgbaColor=[0.0, 0.0, 0.0, 1])
        self.table_start_pos = [0, 0, 0.06]
        self.table_id = p.createMultiBody(1.0, table_shape, table_visual, self.table_start_pos)
        
        # Ball - randomize initial position slightly
        if options and 'ball_start_pos' in options:
            ball_start_pos = options['ball_start_pos']
        else:
            # Random start position within reasonable bounds - 25cm table
            ball_start_pos = [
                np.random.uniform(-0.10, 0.10),  # Stay within 20cm range for safety
                np.random.uniform(-0.10, 0.10),
                0.5
            ]
        
        self.ball_id = p.createMultiBody(
            baseMass=0.1,
            baseCollisionShapeIndex=p.createCollisionShape(p.GEOM_SPHERE, radius=self.ball_radius),
            baseVisualShapeIndex=p.createVisualShape(p.GEOM_SPHERE, radius=self.ball_radius, rgbaColor=[1, 0, 0, 1]),
            basePosition=ball_start_pos
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
        self.table_pitch = np.clip(self.table_pitch, -0.1, 0.1)
        self.table_roll = np.clip(self.table_roll, -0.1, 0.1)
        
        # Update table orientation
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
        
        # Distance from center
        distance_from_center = np.sqrt(ball_x**2 + ball_y**2)
        
        # FIXED REWARD FUNCTION - position must dominate!
        
        # 1. Primary goal: Keep ball at center (MUCH STRONGER)
        # Make position reward the dominant component
        position_reward = max(0.0, 3.0 - distance_from_center / 0.1)  # Much stronger, drops faster
        
        # 2. Penalty for ball falling off
        if distance_from_center > self.table_size:
            return -10.0
        
        # 3. Time bonus - reward for staying on table longer
        time_bonus = 0.1  # Small constant bonus for each step ball stays on table
        
        # 4. Control energy function - but ONLY when ball is reasonably well controlled
        control_energy_reward = 0.0
        if distance_from_center < 0.10:  # Only reward good control when ball is well within table bounds
            # Calculate what the optimal action SHOULD be based on ball state
            optimal_action = self._calculate_optimal_action(ball_x, ball_y, ball_vx, ball_vy)
            
            # Measure how close the agent's action is to the optimal action
            action_error = np.linalg.norm(action - optimal_action)
            
            # Reward inverse of error - closer to optimal = higher reward (but smaller max)
            control_energy_reward = max(0.0, 0.3 - action_error * 3.0)  # Reduced max reward
        
        # 5. Smart velocity penalty (context-dependent)
        velocity_magnitude = np.sqrt(ball_vx**2 + ball_vy**2)
        if distance_from_center < 0.05:  # Ball is close to center
            # When centered, we want it stationary
            velocity_penalty = -velocity_magnitude * 1.0  # Penalize movement when centered
        else:
            # When off-center, don't penalize movement (we want it to move toward center)
            velocity_penalty = 0.0
        
        # 6. Penalty for extreme table angles (discourages big tilts for circular motion)
        table_angle_magnitude = np.sqrt(table_pitch**2 + table_roll**2)
        table_penalty = -table_angle_magnitude * 3.0  # Penalize large table tilts
        
        # 7. Penalty for action oscillation AND circular patterns
        oscillation_penalty = 0.0
        if len(self.prev_actions) >= 6:  # Need more actions to detect circular patterns
            # Check for alternating pattern over last 4 actions (bang-bang detection)
            recent_actions = self.prev_actions[-4:] + [action]  # Last 4 + current = 5 actions
            
            # Look for alternating signs in each action component with LARGE magnitudes
            # Only penalize if the oscillations are also large (bang-bang style)
            pitch_large = [np.sign(a[0]) for a in recent_actions if abs(a[0]) > 0.025]  # Only large actions
            roll_large = [np.sign(a[1]) for a in recent_actions if abs(a[1]) > 0.025]   # Only large actions
            
            # Check if we have alternating signs (oscillation pattern)
            def is_alternating(signs):
                if len(signs) < 3:
                    return False
                alternations = 0
                for i in range(1, len(signs)):
                    if signs[i] != signs[i-1]:
                        alternations += 1
                return alternations >= 2  # At least 2 sign changes = oscillation
            
            # Only penalize if LARGE actions are oscillating (bang-bang)
            if is_alternating(pitch_large) or is_alternating(roll_large):
                oscillation_penalty += -0.5  # Penalty for large oscillations
                
            # NEW: Detect circular/orbital motion patterns
            longer_actions = self.prev_actions[-6:] + [action]  # Last 6 + current = 7 actions
            if len(longer_actions) >= 6:
                # Calculate action magnitudes - circular motion uses consistently large actions
                action_mags = [np.linalg.norm(a) for a in longer_actions]
                avg_magnitude = np.mean(action_mags)
                
                # Check if actions are consistently large (indicating orbital motion)
                if avg_magnitude > 0.02:  # Actions are substantial
                    # Check if we're doing circular motion by looking at action direction changes
                    angles = []
                    for a in longer_actions:
                        if np.linalg.norm(a) > 0.01:  # Only consider significant actions
                            angle = np.arctan2(a[1], a[0])  # Angle of action vector
                            angles.append(angle)
                    
                    if len(angles) >= 5:
                        # Check for smooth directional progression (circular motion)
                        angle_diffs = []
                        for i in range(1, len(angles)):
                            diff = angles[i] - angles[i-1]
                            # Normalize angle difference to [-pi, pi]
                            while diff > np.pi:
                                diff -= 2*np.pi
                            while diff < -np.pi:
                                diff += 2*np.pi
                            angle_diffs.append(abs(diff))
                        
                        # If angle changes are consistent and moderate, it's circular motion
                        avg_angle_change = np.mean(angle_diffs)
                        if 0.3 < avg_angle_change < 2.0:  # Consistent moderate turning
                            oscillation_penalty += -0.8  # Strong penalty for circular motion
                
        # Also penalize if current action is roughly opposite to previous (immediate detection)
        # But only if both actions are reasonably large
        if self.prev_action is not None:
            if (np.linalg.norm(action) > 0.02 and np.linalg.norm(self.prev_action) > 0.02):
                action_dot_product = np.dot(action, self.prev_action)
                if action_dot_product < -0.001:  # Actions are opposite
                    oscillation_penalty += -0.3  # Small additional penalty
        
        # Update previous action for future use
        self.prev_action = action.copy()
        
        # Position-dominated combination - position reward is now 10x more important
        total_reward = position_reward + time_bonus + control_energy_reward + velocity_penalty + table_penalty + oscillation_penalty
        
        return total_reward
    
    def _calculate_optimal_action(self, ball_x, ball_y, ball_vx, ball_vy):
        """Calculate the theoretically optimal action for current ball state"""
        # Simple PD controller logic: proportional to position + derivative (velocity)
        
        # Proportional gains (how much to tilt based on position error)
        kp = 0.8  # Position gain
        kd = 0.3  # Velocity (derivative) gain
        
        # Calculate desired table tilts to center the ball
        # Tilt table opposite to ball position to "roll" ball toward center
        desired_pitch = -ball_x * kp - ball_vx * kd  # Negative because we want to oppose ball movement
        desired_roll = -ball_y * kp - ball_vy * kd
        
        # Clip to action space limits
        desired_pitch = np.clip(desired_pitch, -0.05, 0.05)
        desired_roll = np.clip(desired_roll, -0.05, 0.05)
        
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
