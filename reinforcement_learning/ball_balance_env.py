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
    
    def __init__(self, render_mode="human", max_steps=2000, control_freq=50, enable_servo_uncertainty=True):
        super().__init__()
        
        self.render_mode = render_mode
        self.max_steps = max_steps
        self.current_step = 0
        self.enable_servo_uncertainty = enable_servo_uncertainty
        
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
        
        # 🔧 SERVO UNCERTAINTY MODEL - Realistic XL430-250T servo behavior for RL training
        # Parameters tuned for Dynamixel XL430-250T servos (high-quality, 60Hz update rate)
        self.servo_uncertainty = {
            'enable': self.enable_servo_uncertainty,  # Can be disabled for perfect control debugging
            
            # Backlash/Dead Zone - XL430 has minimal backlash but still present
            'backlash_degrees': 0.05,  # ±0.05° dead zone (XL430 is quite precise)
            
            # Response delay - 60Hz servo update rate = ~16.7ms delay
            'response_delay_steps': 1,  # ~20ms delay at 50Hz control (60Hz servo = minimal delay)
            
            # Position noise - XL430 has good resolution but still has jitter
            'position_noise_std': 0.02,  # 0.02° standard deviation (high-quality servo)
            
            # Saturation effects - XL430 has smooth response curves
            'saturation_softness': 0.9,  # Very gradual saturation (quality servo)
            
            # Hysteresis - minimal for XL430 due to quality gears
            'hysteresis_strength': 0.01,  # 0.01° hysteresis effect (very low)
            
            # Compliance - XL430 is quite stiff but table weight still affects it slightly
            'compliance_factor': 0.005,  # 0.5% position error under load (stiff servo)
        }
        
        # Servo state tracking for uncertainty model
        self.servo_state = {
            'commanded_pitch': 0.0,      # What we told servo to do
            'commanded_roll': 0.0,
            'actual_pitch': 0.0,         # What servo actually achieved (with uncertainty)
            'actual_roll': 0.0,
            'previous_pitch': 0.0,       # For hysteresis calculation
            'previous_roll': 0.0,
            'delay_buffer_pitch': [],    # Command delay buffer
            'delay_buffer_roll': [],
            'last_direction_pitch': 0,   # -1, 0, 1 for hysteresis
            'last_direction_roll': 0,
        }
        
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
        
        # 🔧 Reset servo uncertainty state
        self.servo_state = {
            'commanded_pitch': 0.0,
            'commanded_roll': 0.0,
            'actual_pitch': 0.0,
            'actual_roll': 0.0,
            'previous_pitch': 0.0,
            'previous_roll': 0.0,
            'delay_buffer_pitch': [],
            'delay_buffer_roll': [],
            'last_direction_pitch': 0,
            'last_direction_roll': 0,
        }
        
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
        
        # Table - sleek dark surface with slight reflection
        table_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[self.table_size, self.table_size, 0.004])
        table_visual = p.createVisualShape(p.GEOM_BOX, halfExtents=[self.table_size, self.table_size, 0.004], 
                                         rgbaColor=[0.1, 0.1, 0.1, 1],  # Dark surface
                                         specularColor=[0.2, 0.2, 0.2])  # Slight metallic look
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
        
        self.ball_id = p.createMultiBody(
            baseMass=0.0027,  # 2.7 grams in kg (realistic ping pong ball)
            baseCollisionShapeIndex=p.createCollisionShape(p.GEOM_SPHERE, radius=self.ball_radius),
            baseVisualShapeIndex=p.createVisualShape(p.GEOM_SPHERE, radius=self.ball_radius, 
                                                   rgbaColor=[0.9, 0.9, 0.9, 1],      # Bright white
                                                   specularColor=[0.8, 0.8, 0.8]),    # Shiny surface
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
        
        # 🔧 Apply servo uncertainty to get realistic actual table angles
        actual_pitch, actual_roll = self.apply_servo_uncertainty(self.table_pitch, self.table_roll)
        
        # Update table orientation with ACTUAL servo positions (including uncertainty)
        quat = p.getQuaternionFromEuler([actual_pitch, actual_roll, 0])
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
    
    def apply_servo_uncertainty(self, commanded_pitch, commanded_roll):
        """
        Apply realistic servo uncertainty and mechanical imperfections
        
        This simulates real-world XL430-250T servo behavior including:
        - Backlash/dead zone
        - Response delays
        - Position noise
        - Saturation effects
        - Hysteresis
        - Mechanical compliance
        
        Returns actual achieved angles (with uncertainty)
        """
        if not self.servo_uncertainty['enable']:
            return commanded_pitch, commanded_roll
        
        # 1. RESPONSE DELAY - Commands take time to execute
        # Add current commands to delay buffer
        self.servo_state['delay_buffer_pitch'].append(commanded_pitch)
        self.servo_state['delay_buffer_roll'].append(commanded_roll)
        
        # Keep buffer at correct size
        delay_steps = self.servo_uncertainty['response_delay_steps']
        if len(self.servo_state['delay_buffer_pitch']) > delay_steps:
            self.servo_state['delay_buffer_pitch'].pop(0)
            self.servo_state['delay_buffer_roll'].pop(0)
        
        # Use delayed command if buffer is full, otherwise use current
        if len(self.servo_state['delay_buffer_pitch']) >= delay_steps:
            delayed_pitch = self.servo_state['delay_buffer_pitch'][0]
            delayed_roll = self.servo_state['delay_buffer_roll'][0]
        else:
            delayed_pitch = commanded_pitch
            delayed_roll = commanded_roll
        
        # 2. BACKLASH/DEAD ZONE - Must overcome mechanical slack
        backlash_rad = np.radians(self.servo_uncertainty['backlash_degrees'])
        
        def apply_backlash(commanded, actual, previous):
            movement = commanded - previous
            if abs(movement) < backlash_rad:
                # Movement too small to overcome backlash - no change
                return actual
            else:
                # Movement large enough - but reduced by backlash amount
                if movement > 0:
                    return actual + max(0, movement - backlash_rad)
                else:
                    return actual + min(0, movement + backlash_rad)
        
        pitch_after_backlash = apply_backlash(
            delayed_pitch, 
            self.servo_state['actual_pitch'], 
            self.servo_state['previous_pitch']
        )
        roll_after_backlash = apply_backlash(
            delayed_roll, 
            self.servo_state['actual_roll'], 
            self.servo_state['previous_roll']
        )
        
        # 3. HYSTERESIS - Different behavior based on movement direction
        hysteresis = np.radians(self.servo_uncertainty['hysteresis_strength'])
        
        def apply_hysteresis(target, actual, last_direction):
            movement_direction = np.sign(target - actual)
            
            if movement_direction != last_direction and movement_direction != 0:
                # Direction changed - apply hysteresis offset
                if movement_direction > 0:
                    return target - hysteresis
                else:
                    return target + hysteresis
            return target
        
        # Update movement directions
        pitch_direction = np.sign(delayed_pitch - self.servo_state['actual_pitch'])
        roll_direction = np.sign(delayed_roll - self.servo_state['actual_roll'])
        
        pitch_with_hysteresis = apply_hysteresis(
            pitch_after_backlash, 
            self.servo_state['actual_pitch'],
            self.servo_state['last_direction_pitch']
        )
        roll_with_hysteresis = apply_hysteresis(
            roll_after_backlash, 
            self.servo_state['actual_roll'],
            self.servo_state['last_direction_roll']
        )
        
        # 4. SATURATION EFFECTS - Non-linear response near limits
        servo_limit = 0.0559  # ±3.2° servo limit
        softness = self.servo_uncertainty['saturation_softness']
        
        def soft_saturation(angle, limit, softness):
            """Smooth saturation instead of hard clipping"""
            if abs(angle) < limit * softness:
                return angle  # Linear region
            else:
                # Soft saturation region
                sign = np.sign(angle)
                excess = abs(angle) - limit * softness
                max_excess = limit * (1 - softness)
                # Smooth transition using tanh
                return sign * (limit * softness + max_excess * np.tanh(excess / max_excess))
        
        pitch_saturated = soft_saturation(pitch_with_hysteresis, servo_limit, softness)
        roll_saturated = soft_saturation(roll_with_hysteresis, servo_limit, softness)
        
        # 5. COMPLIANCE - Servo "gives" under load (table weight)
        compliance = self.servo_uncertainty['compliance_factor']
        pitch_compliant = pitch_saturated * (1 - compliance)
        roll_compliant = roll_saturated * (1 - compliance)
        
        # 6. POSITION NOISE - Random deviations
        noise_std = np.radians(self.servo_uncertainty['position_noise_std'])
        pitch_noise = np.random.normal(0, noise_std)
        roll_noise = np.random.normal(0, noise_std)
        
        # Final actual positions
        actual_pitch = pitch_compliant + pitch_noise
        actual_roll = roll_compliant + roll_noise
        
        # Update servo state for next iteration
        self.servo_state['commanded_pitch'] = commanded_pitch
        self.servo_state['commanded_roll'] = commanded_roll
        self.servo_state['previous_pitch'] = self.servo_state['actual_pitch']
        self.servo_state['previous_roll'] = self.servo_state['actual_roll']
        self.servo_state['actual_pitch'] = actual_pitch
        self.servo_state['actual_roll'] = actual_roll
        self.servo_state['last_direction_pitch'] = pitch_direction
        self.servo_state['last_direction_roll'] = roll_direction
        
        return actual_pitch, actual_roll
    
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
        kp = 0.8  # Position gain
        kd = 0.08  # Velocity (derivative) gain
        
        # Calculate desired table tilts to center the ball
        # Tilt table opposite to ball position to "roll" ball toward center
        desired_pitch = -ball_x * kp - ball_vx * kd  # Negative because we want to oppose ball movement
        desired_roll = -ball_y * kp - ball_vy * kd
        
        # Clip to action space limits
        desired_pitch = np.clip(desired_pitch, -0.0524, 0.0524)
        desired_roll = np.clip(desired_roll, -0.0524, 0.0524)
        
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
