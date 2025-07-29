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
        self.observation_space = spaces.Box(
            low=np.array([-0.3, -0.3, -2.0, -2.0, -0.1, -0.1]),
            high=np.array([0.3, 0.3, 2.0, 2.0, 0.1, 0.1]),
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
        self.table_size = 0.25
        
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
            # Random start position within reasonable bounds
            ball_start_pos = [
                np.random.uniform(-0.15, 0.15),
                np.random.uniform(-0.15, 0.15),
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
        
        # TUNED REWARD FUNCTION - focus on core objectives with better scaling
        
        # 1. Primary goal: Keep ball at center (exponential for better gradient)
        position_reward = np.exp(-distance_from_center * 8.0)  # Strong reward near center, decays exponentially
        
        # 2. Secondary goal: Minimize velocity (quadratic penalty for smoother control)
        velocity_magnitude = np.sqrt(ball_vx**2 + ball_vy**2)
        velocity_reward = np.exp(-velocity_magnitude * 3.0)  # Exponential decay for velocity
        
        # 3. Tertiary goal: Minimize plate movement (adjusted scale)
        action_magnitude = np.sqrt(action[0]**2 + action[1]**2)
        action_reward = 1.0 - min(action_magnitude / 0.05, 1.0)  # Use full action range for scaling
        
        # 4. Severe penalty for ball falling off (increased penalty)
        if distance_from_center > self.table_size:
            return -20.0  # Increased from -10 for stronger avoidance
        
        # 5. Penalty for sudden servo movements (jerky control) - tuned thresholds
        jerk_penalty = 0.0
        if self.prev_action is not None:
            delta_pitch = abs(action[0] - self.prev_action[0])
            delta_roll = abs(action[1] - self.prev_action[1])
            jerk_magnitude = np.sqrt(delta_pitch**2 + delta_roll**2)
            jerk_penalty = -min(jerk_magnitude / 0.02, 1.0) * 0.8  # Reduced max penalty and increased threshold

        # 6. Penalize bang-bang control (tuned thresholds)
        bang_bang_penalty = 0.0
        action_limit = 0.05  # Your action space limit
        threshold = 0.04     # 80% of the limit (reduced from 90%)
        
        # Progressive penalty instead of hard cutoff
        for action_component in [action[0], action[1]]:
            if abs(action_component) > threshold:
                excess = (abs(action_component) - threshold) / (action_limit - threshold)
                bang_bang_penalty -= excess**2 * 1.5  # Quadratic penalty, max -1.5 per component
        
        # Update previous action
        self.prev_action = action.copy()  # Use copy to avoid reference issues
        
        # Combine rewards with tuned weights
        total_reward = (position_reward * 3.0 +    # Position is most important (increased weight)
                       velocity_reward * 1.5 +    # Velocity is very important for stability
                       action_reward * 0.3 +      # Action efficiency is less important
                       jerk_penalty +             # Jerk penalty (reduced impact)
                       bang_bang_penalty)        # Bang-bang penalty (reduced impact)
        
        # 7. Stability bonus: Extra reward when ball is stable and actions are minimal (tuned)
        if distance_from_center < 0.08 and velocity_magnitude < 0.15:  # Relaxed thresholds
            stability_multiplier = 1.0 - (distance_from_center / 0.08)  # Scale bonus by how close to center
            velocity_multiplier = 1.0 - (velocity_magnitude / 0.15)     # Scale bonus by how slow the ball is
            stability_reward = 1.5 * stability_multiplier * velocity_multiplier * (1.0 - action_magnitude / 0.05)
            total_reward += stability_reward
        
        return total_reward
    
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
