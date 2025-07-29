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
    - Ball position (x, y)
    - Ball velocity (vx, vy)
    - Table angles (pitch, roll)
    
    Action Space:
    - Table pitch angle change
    - Table roll angle change
    """
    
    def __init__(self, render_mode="human", max_steps=1000):
        super().__init__()
        
        self.render_mode = render_mode
        self.max_steps = max_steps
        self.current_step = 0
        
        # Action space: pitch and roll angle changes (in radians)
        self.action_space = spaces.Box(
            low=-0.05, high=0.05, shape=(2,), dtype=np.float32
        )
        
        # Observation space: [ball_x, ball_y, ball_vx, ball_vy, table_pitch, table_roll]
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
        self.prev_ball_pos = None
        
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
        self.prev_ball_pos = None
        
        # Disconnect existing physics client if it exists
        if self.physics_client is not None:
            p.disconnect(self.physics_client)
        
        # Initialize PyBullet
        if self.render_mode == "human":
            self.physics_client = p.connect(p.GUI)
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
        reward = self._calculate_reward(observation)
        
        # Check termination conditions
        terminated = self._is_terminated(observation)
        truncated = self.current_step >= self.max_steps
        
        info = self._get_info()
        
        return observation, reward, terminated, truncated, info
    
    def _get_observation(self):
        # Ball position and orientation
        ball_pos, _ = p.getBasePositionAndOrientation(self.ball_id)
        ball_x, ball_y, ball_z = ball_pos
        
        # Ball velocity
        ball_vel, _ = p.getBaseVelocity(self.ball_id)
        ball_vx, ball_vy, _ = ball_vel
        
        # Store for velocity calculation
        if self.prev_ball_pos is None:
            self.prev_ball_pos = [ball_x, ball_y]
        
        observation = np.array([
            ball_x, ball_y, ball_vx, ball_vy, self.table_pitch, self.table_roll
        ], dtype=np.float32)
        
        self.prev_ball_pos = [ball_x, ball_y]
        
        return observation
    
    def _calculate_reward(self, observation):
        ball_x, ball_y, ball_vx, ball_vy, table_pitch, table_roll = observation
        
        # Distance from center
        distance_from_center = np.sqrt(ball_x**2 + ball_y**2)
        
        # Reward components - BALANCED VERSION
        # 1. Distance reward (exponential decay, but scaled down)
        position_reward = np.exp(-distance_from_center * 3.0) * 0.5  # Max 0.5 at center
        
        # 2. Linear distance penalty (gentle slope)
        position_penalty = -distance_from_center * 2.0
        
        # 3. Small penalty for high velocity (encourage stability)
        velocity_penalty = -(abs(ball_vx) + abs(ball_vy)) * 0.2
        
        # 4. Small penalty for large table angles (encourage efficiency)
        angle_penalty = -(abs(table_pitch) + abs(table_roll)) * 0.5
        
        # 5. Small bonus for being very close to center
        if distance_from_center < 0.05:
            position_reward += 0.5  # Reduced from 2.0
        
        # 6. Large penalty if ball falls off table
        if distance_from_center > self.table_size:
            return -5.0  # Moderate penalty to avoid huge spikes
        
        # 7. Small time bonus for surviving each step
        time_bonus = 0.01
        
        total_reward = position_reward + position_penalty + velocity_penalty + angle_penalty + time_bonus
        
        return total_reward
    
    def _is_terminated(self, observation):
        ball_x, ball_y, _, _, _, _ = observation
        
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
