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
    
    def __init__(self, render_mode="human", max_steps=2000):
        super().__init__()
        
        self.render_mode = render_mode
        self.max_steps = max_steps
        self.current_step = 0
        
        # Action space: pitch and roll angle changes (in radians)
        self.action_space = spaces.Box(
            low=-0.05, high=0.05, shape=(2,), dtype=np.float32
        )
        
        # Observation space: [ball_x, ball_y, ball_vx, ball_vy, table_pitch, table_roll] - ESTIMATED VELOCITY
        self.observation_space = spaces.Box(
            low=np.array([-0.3, -0.3, -2.0, -2.0, -0.1, -0.1]),
            high=np.array([0.3, 0.3, 2.0, 2.0, 0.1, 0.1]),
            dtype=np.float32
        )
        
        # Physics parameters
        self.dt = 1.0 / 240.0
        self.ball_radius = 0.02
        self.table_size = 0.25
        
        # State tracking
        self.table_pitch = 0.0
        self.table_roll = 0.0
        self.prev_ball_pos = None  # For velocity estimation
        self.prev_table_pitch = 0.0
        self.prev_table_roll = 0.0
        self.prev_actions = []  # Track action history for oscillation detection
        
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
        self.prev_table_pitch = 0.0
        self.prev_table_roll = 0.0
        self.prev_actions = []  # Reset action history
        
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
        
        # Track action history for oscillation detection (keep last 4 actions)
        self.prev_actions.append(action.copy())
        if len(self.prev_actions) > 4:
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
        
        # Step simulation
        p.stepSimulation()
        if self.render_mode == "human":
            time.sleep(self.dt)
        
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
        
        # Estimate velocity from position differences (realistic approach)
        ball_vx, ball_vy = 0.0, 0.0
        if self.prev_ball_pos is not None:
            ball_vx = (ball_x - self.prev_ball_pos[0]) / self.dt
            ball_vy = (ball_y - self.prev_ball_pos[1]) / self.dt
        
        # Update previous position for next velocity calculation
        self.prev_ball_pos = [ball_x, ball_y]
        
        # Return position + estimated velocity + table angles
        observation = np.array([
            ball_x, ball_y, ball_vx, ball_vy, self.table_pitch, self.table_roll
        ], dtype=np.float32)
        
        return observation
    
    def _calculate_reward(self, observation, action):
        ball_x, ball_y, ball_vx, ball_vy, table_pitch, table_roll = observation  # Updated for 6D format with estimated velocity
        
        # Velocity is now included in observation (estimated from position differences)
        
        # Update previous position for next step
        self.prev_ball_pos = [ball_x, ball_y]
        
        # Distance from center
        distance_from_center = np.sqrt(ball_x**2 + ball_y**2)
        
        # Reward components - POSITION ONLY like PID, but still reward stability
        # 1. Strong reward for keeping ball close to center
        position_reward = np.exp(-distance_from_center * 5.0) * 2.0
        
        # 2. Moderate distance penalty
        position_penalty = -distance_from_center * 1.5
        
        # 3. Reward for low estimated velocity (stability) - calculated from position
        velocity_magnitude = np.sqrt(ball_vx**2 + ball_vy**2)
        velocity_reward = np.exp(-velocity_magnitude * 2.0) * 0.5
        
        # 4. Moderate penalty for large table angles
        angle_penalty = -(abs(table_pitch) + abs(table_roll)) * 0.3
        
        # 5. Light penalty for large actions (servo-friendly)
        action_magnitude_penalty = -(abs(action[0]) + abs(action[1])) * 0.5
        
        # 6. Light penalty for rapid angle changes (smooth movement)
        angle_change_pitch = abs(table_pitch - self.prev_table_pitch)
        angle_change_roll = abs(table_roll - self.prev_table_roll)
        smooth_movement_penalty = -(angle_change_pitch + angle_change_roll) * 1.0
        
        # 7. Special reward for being very close to center AND stable
        if distance_from_center < 0.05 and velocity_magnitude < 0.1:
            stability_bonus = 1.0
        else:
            stability_bonus = 0.0
        
        # 8. STRONG penalty for oscillating behavior (detect sign flipping)
        oscillation_penalty = 0.0
        if len(self.prev_actions) >= 3:  # Need at least 3 actions to detect oscillation
            # Check for sign flipping in recent actions
            recent_actions = np.array(self.prev_actions[-3:])  # Last 3 actions
            
            # Detect if actions are alternating signs (oscillating)
            pitch_signs = np.sign(recent_actions[:, 0])
            roll_signs = np.sign(recent_actions[:, 1])
            
            # Check for alternating pattern: +, -, + or -, +, -
            pitch_oscillating = (pitch_signs[0] != pitch_signs[1] and pitch_signs[1] != pitch_signs[2])
            roll_oscillating = (roll_signs[0] != roll_signs[1] and roll_signs[1] != roll_signs[2])
            
            # Check for maximum magnitude oscillations (the worst kind)
            max_actions = np.abs(recent_actions) > 0.04  # Near maximum action
            
            if (pitch_oscillating or roll_oscillating) and np.any(max_actions):
                oscillation_penalty = -2.0  # Heavy penalty for oscillating at max
                
        # 9. Reward for action consistency (small, similar actions over time)
        action_consistency_bonus = 0.0
        if len(self.prev_actions) >= 2:
            # Reward for making similar actions (consistent control)
            action_similarity = 1.0 - np.mean(np.abs(action - self.prev_actions[-1]))
            action_consistency_bonus = action_similarity * 0.3
            
        # 10. Special penalty for using maximum actions repeatedly
        max_action_penalty = 0.0
        if np.any(np.abs(action) > 0.045):  # Close to maximum action space
            max_action_penalty = -0.5
        
        # 11. Large penalty if ball falls off table
        if distance_from_center > self.table_size:
            return -10.0
        
        # 12. Time bonus for surviving each step
        time_bonus = 0.02
        
        total_reward = (position_reward + position_penalty + velocity_reward + 
                       angle_penalty + action_magnitude_penalty + smooth_movement_penalty + 
                       stability_bonus + oscillation_penalty + action_consistency_bonus + 
                       max_action_penalty + time_bonus)
        
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
