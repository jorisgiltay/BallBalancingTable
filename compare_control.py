import pybullet as p
import pybullet_data
import time
import numpy as np
from pid_controller import PIDController
import argparse
import os
import threading
import queue


class BallBalanceComparison:
    """
    Compare PID control vs Reinforcement Learning control
    """
    
    def __init__(self, control_method="pid", control_freq=50, enable_visuals=False):
        self.control_method = control_method
        self.control_freq = control_freq  # Hz - control update frequency (default 50 Hz like servos)
        self.enable_visuals = enable_visuals  # Flag to enable/disable visual indicators
        
        # Timing parameters
        self.physics_freq = 240  # Hz - physics simulation frequency
        self.physics_dt = 1.0 / self.physics_freq  # Physics timestep
        self.control_dt = 1.0 / self.control_freq  # Control timestep
        self.physics_steps_per_control = self.physics_freq // self.control_freq  # Steps per control update
        
        # PID controllers (create these BEFORE setup_simulation)
        self.pitch_pid = PIDController(kp=10.0, ki=0.1, kd=2.0, output_limits=(-0.05, 0.05))
        self.roll_pid = PIDController(kp=10.0, ki=0.1, kd=2.0, output_limits=(-0.05, 0.05))
        
        # Thread-safe visual system with optimized queue
        if self.enable_visuals:
            self.visual_queue = queue.Queue(maxsize=5)  # Small buffer to keep data fresh
            self.visual_thread_running = True
            self.visual_thread = threading.Thread(target=self._visual_matplotlib_thread, daemon=True)
        
        self.setup_simulation()
        
        # Start visual thread if enabled
        if self.enable_visuals:
            self.visual_thread.start()
        
        # RL model (will be loaded if using RL)
        self.rl_model = None
        if control_method == "rl":
            self.load_rl_model()
        
        # State tracking
        self.table_pitch = 0.0
        self.table_roll = 0.0
        self.prev_ball_pos = None  # For velocity estimation  
        self.step_count = 0
        self.randomize_ball = True  # Start with randomized positions
        
    def setup_simulation(self):
        """Initialize PyBullet simulation"""
        p.connect(p.GUI)
        
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
        
        # Ball
        self.ball_radius = 0.02
        self.reset_ball()
        
    def reset_ball(self, position=None, randomize=True):
        """Reset ball to initial position"""
        if position is None:
            if randomize:
                # Random position like in RL training - adjusted for 25cm table
                position = [
                    np.random.uniform(-0.11, 0.11),  # Stay within 22cm range for safety
                    np.random.uniform(-0.11, 0.11),
                    0.5
                ]
                print(f"Ball reset to random position: ({position[0]:.2f}, {position[1]:.2f})")
            else:
                # Fixed position for consistent testing - well within 25cm table
                position = [0.06, 0.08, 0.5]  # Distance = 0.10m (safely within 0.125m table)
                print(f"Ball reset to fixed position: ({position[0]:.2f}, {position[1]:.2f})")
            
        if hasattr(self, 'ball_id'):
            p.removeBody(self.ball_id)
            
        self.ball_id = p.createMultiBody(
            baseMass=0.0027,  # 2.7 grams in kg
            baseCollisionShapeIndex=p.createCollisionShape(p.GEOM_SPHERE, radius=self.ball_radius),
            baseVisualShapeIndex=p.createVisualShape(p.GEOM_SPHERE, radius=self.ball_radius, 
                                                   rgbaColor=[0.9, 0.9, 0.9, 1],      # Bright white
                                                   specularColor=[0.8, 0.8, 0.8]),    # Shiny surface
            basePosition=position
        )
        
        # Reset PID controllers
        self.pitch_pid.reset()
        self.roll_pid.reset()
        self.table_pitch = 0.0
        self.table_roll = 0.0
        self.prev_ball_pos = None  # Reset velocity estimation
        self.step_count = 0
        
        # Let ball settle
        for _ in range(100):
            p.stepSimulation()
    
    def _visual_matplotlib_thread(self):
        """Real-time matplotlib dashboard - thread-safe and professional"""
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        from matplotlib.animation import FuncAnimation
        
        # Set up the matplotlib figure with simpler layout for performance
        plt.style.use('dark_background')
        fig = plt.figure(figsize=(6, 8))
        fig.suptitle('Ball Balance Dashboard', fontsize=14, fontweight='bold', color='white')
        
        # Create simpler layout with fewer subplots
        gs = fig.add_gridspec(3, 2, height_ratios=[1.5, 1, 1], hspace=0.4, wspace=0.3)
        
        # Ball position plot (top-down view of table)
        ax_table = fig.add_subplot(gs[0, :])
        ax_table.set_xlim(-0.15, 0.15)
        ax_table.set_ylim(-0.15, 0.15)
        ax_table.set_aspect('equal')
        ax_table.set_title('Ball Position on Table', fontweight='bold')
        ax_table.set_xlabel('X Position (m)')
        ax_table.set_ylabel('Y Position (m)')
        ax_table.grid(True, alpha=0.3)
        
        # Draw table boundary
        table_circle = plt.Circle((0, 0), 0.125, fill=False, color='white', linewidth=2)
        ax_table.add_patch(table_circle)
        
        # Ball position marker
        ball_marker, = ax_table.plot(0, 0, 'o', color='red', markersize=8)
        
        # Control effort and angles combined
        ax_control = fig.add_subplot(gs[1, :])
        ax_control.set_title('Control Actions & Table Angles', fontweight='bold')
        ax_control.set_ylim(-0.06, 0.06)
        
        # Combined bars for actions and angles
        x_pos = np.arange(4)
        labels = ['Action\nPitch', 'Action\nRoll', 'Table\nPitch', 'Table\nRoll']
        combined_bars = ax_control.bar(x_pos, [0, 0, 0, 0], 
                                     color=['cyan', 'orange', 'lightblue', 'lightyellow'])
        ax_control.set_xticks(x_pos)
        ax_control.set_xticklabels(labels, fontsize=9)
        ax_control.set_ylabel('Angle (rad)')
        ax_control.grid(True, alpha=0.3)
        
        # Distance and status combined
        ax_status = fig.add_subplot(gs[2, :])
        ax_status.set_title('Performance Status', fontweight='bold')
        ax_status.axis('off')
        
        # Single text display for all metrics (more efficient)
        status_text = ax_status.text(0.05, 0.7, '', fontsize=11, color='white', 
                                   verticalalignment='top', fontfamily='monospace')
        
        # Animation update function - optimized for smooth performance with blitting
        def update_dashboard(frame):
            try:
                # Get data from queue (non-blocking)
                data = self.visual_queue.get_nowait()
                
                # Update ball position (most frequent update)
                ball_marker.set_data([data['ball_x']], [data['ball_y']])
                
                # Update combined bars (actions and angles) - only if changed significantly
                heights = [data['action'][0], data['action'][1], data['pitch'], data['roll']]
                for i, (bar, height) in enumerate(zip(combined_bars, heights)):
                    bar.set_height(height)
                    
                    # Color based on magnitude
                    if i < 2:  # Action bars
                        color = 'red' if abs(height) > 0.03 else ('cyan' if i == 0 else 'orange')
                    else:  # Angle bars  
                        color = 'red' if abs(height) > 0.05 else ('lightblue' if i == 2 else 'lightyellow')
                    bar.set_color(color)
                
                # Update text only occasionally to reduce rendering load
                if frame % 3 == 0:  # Update text every 3rd frame (still smooth but less overhead)
                    velocity_mag = np.sqrt(data['ball_vx']**2 + data['ball_vy']**2)
                    action_mag = np.linalg.norm(data['action'])
                    
                    # Choose status color based on performance
                    if data['distance'] < 0.05:
                        text_color = 'green'
                    elif data['distance'] < 0.1:
                        text_color = 'yellow'
                    else:
                        text_color = 'red'
                    
                    status_text.set_text(
                        f"Control: {data['method'].upper():<3} | Step: {data['step']:<6} | Distance: {data['distance']:.3f}m\n"
                        f"Ball Pos: ({data['ball_x']:+.3f}, {data['ball_y']:+.3f}) | Velocity: {velocity_mag:.3f} m/s\n"
                        f"Action: [{data['action'][0]:+.3f}, {data['action'][1]:+.3f}] | Magnitude: {action_mag:.4f}\n"
                        f"Table: Pitch {data['pitch']:+.3f} | Roll {data['roll']:+.3f}"
                    )
                    status_text.set_color(text_color)
                
            except queue.Empty:
                pass  # No new data, keep current display
            except Exception as e:
                print(f"Dashboard update error: {e}")
            
            # Return all artists for blitting (much faster rendering)
            return [ball_marker] + list(combined_bars) + [status_text]
        
        # Set up animation with optimized update rate for smoothness vs performance
        ani = FuncAnimation(fig, update_dashboard, interval=100, blit=True, cache_frame_data=False)
        
        # Show the dashboard
        plt.tight_layout()
        plt.show(block=True)  # Block to keep window open
        
        print("Dashboard window closed")
    
    def _update_visual_data(self, ball_x, ball_y, ball_vx, ball_vy, distance, control_action, step):
        """Send data to visual thread via queue (completely thread-safe)"""
        if self.enable_visuals:
            try:
                data = {
                    'method': self.control_method,
                    'step': step,
                    'ball_x': ball_x,
                    'ball_y': ball_y,
                    'ball_vx': ball_vx,
                    'ball_vy': ball_vy,
                    'distance': distance,
                    'pitch': self.table_pitch,
                    'roll': self.table_roll,
                    'action': control_action if control_action else [0.0, 0.0]
                }
                # Try to put data, if queue is full, remove oldest and add new
                try:
                    self.visual_queue.put_nowait(data)
                except queue.Full:
                    try:
                        self.visual_queue.get_nowait()  # Remove oldest
                        self.visual_queue.put_nowait(data)  # Add new
                    except queue.Empty:
                        pass  # Queue became empty, just skip
            except Exception:
                pass  # Skip any queue errors to avoid affecting simulation
    
    def load_rl_model(self):
        """Load trained RL model"""
        try:
            from stable_baselines3 import PPO
            
            # Try multiple model paths
            model_paths = [
                "models/best_model"
            ]
            
            for model_path in model_paths:
                full_path = model_path + ".zip"
                print(f"Checking for model at: {full_path}")
                if os.path.exists(full_path):
                    try:
                        self.rl_model = PPO.load(model_path)
                        print(f"✅ RL model loaded successfully from {model_path}")
                        return
                    except Exception as e:
                        print(f"❌ Error loading model from {model_path}: {e}")
                        continue
                else:
                    print(f"❌ Model file not found: {full_path}")
            
            # If we get here, no model was loaded
            print("❌ No RL models found. Available files in models/:")
            if os.path.exists("models"):
                for file in os.listdir("models"):
                    print(f"   - {file}")
            else:
                print("   - models/ directory doesn't exist")
            
            print("Switching to PID control.")
            self.control_method = "pid"
            
        except ImportError as e:
            print(f"❌ stable_baselines3 not available: {e}")
            print("Switching to PID control.")
            self.control_method = "pid"
        except Exception as e:
            print(f"❌ Unexpected error loading RL model: {e}")
            print("Switching to PID control.")
            self.control_method = "pid"
    
    def get_observation(self):
        """Get current state observation for RL - with estimated velocity using proper control timestep"""
        # Ball position from sensor/camera
        ball_pos, _ = p.getBasePositionAndOrientation(self.ball_id)
        ball_x, ball_y, ball_z = ball_pos
        
        # Estimate velocity from position differences using control timestep
        ball_vx, ball_vy = 0.0, 0.0
        if self.prev_ball_pos is not None:
            # Use control timestep for proper velocity estimation
            ball_vx = (ball_x - self.prev_ball_pos[0]) / self.control_dt
            ball_vy = (ball_y - self.prev_ball_pos[1]) / self.control_dt
        
        # Update previous position for next velocity calculation
        self.prev_ball_pos = [ball_x, ball_y]
        
        # Return position + estimated velocity + table angles (6D like RL environment)
        observation = np.array([
            ball_x, ball_y, ball_vx, ball_vy, self.table_pitch, self.table_roll
        ], dtype=np.float32)
        
        return observation
    
    def pid_control(self, ball_x, ball_y, ball_vx, ball_vy):
        """PID control logic with estimated velocity available (but not used in basic PID)"""
        # Basic PID uses only position error, but velocity is available if needed
        # Use control timestep for PID updates
        pitch_angle = -self.pitch_pid.update(ball_y, self.control_dt)
        roll_angle = self.roll_pid.update(ball_x, self.control_dt)
        return pitch_angle, roll_angle
    
    def rl_control(self, observation):
        """RL control logic"""
        if self.rl_model is None:
            return 0.0, 0.0
        
        action, _ = self.rl_model.predict(observation, deterministic=True)
        # Only print RL actions occasionally to avoid spam
        if self.step_count % (self.control_freq * 2) == 0:  # Print every 2 seconds
            print(f"RL action: {action}")
        return action[0], action[1]  # pitch_change, roll_change
    
    def run_simulation(self):
        """Main simulation loop with configurable control rate"""
        print(f"Running simulation with {self.control_method.upper()} control")
        print(f"Control frequency: {self.control_freq} Hz")
        print(f"Physics frequency: {self.physics_freq} Hz")
        print(f"Physics steps per control: {self.physics_steps_per_control}")
        print("Controls:")
        print("  'r' - Reset ball")
        print("  'f' - Toggle fixed/random ball position")
        print("  'p' - Switch to PID control")
        print("  'l' - Switch to RL control") 
        print("  'q' - Quit")
        print(f"Ball position mode: {'Random' if self.randomize_ball else 'Fixed'}")
        
        physics_step_count = 0  # Track physics steps
        
        while True:
            # Only run control logic at the specified control frequency
            if physics_step_count % self.physics_steps_per_control == 0:
                # Get observation (includes position + estimated velocity)
                observation = self.get_observation()
                ball_x, ball_y, ball_vx, ball_vy = observation[0], observation[1], observation[2], observation[3]
                
                # Get ball height for collision detection
                ball_pos, _ = p.getBasePositionAndOrientation(self.ball_id)
                ball_z = ball_pos[2]
                
                # Control logic - runs at control frequency
                control_action = None
                if self.control_method == "pid":
                    pitch_angle, roll_angle = self.pid_control(ball_x, ball_y, ball_vx, ball_vy)
                    # For PID, these are absolute angles
                    self.table_pitch = pitch_angle
                    self.table_roll = roll_angle
                    control_action = [pitch_angle, roll_angle]
                else:  # RL
                    pitch_change, roll_change = self.rl_control(observation)
                    # For RL, these are angle changes
                    self.table_pitch += pitch_change
                    self.table_roll += roll_change
                    # Clip to reasonable limits
                    self.table_pitch = np.clip(self.table_pitch, -0.1, 0.1)
                    self.table_roll = np.clip(self.table_roll, -0.1, 0.1)
                    control_action = [pitch_change, roll_change]
                
                # Update table orientation
                quat = p.getQuaternionFromEuler([self.table_pitch, self.table_roll, 0])
                p.resetBasePositionAndOrientation(self.table_id, self.table_start_pos, quat)
                
                # Update visual data (thread-safe) - every 10 steps for smoother updates
                if self.step_count % 10 == 0:
                    distance_from_center = np.sqrt(ball_x**2 + ball_y**2)
                    self._update_visual_data(ball_x, ball_y, ball_vx, ball_vy, distance_from_center, control_action, self.step_count)
                
                # Check if ball fell off - 25cm table (radius = 0.125m)
                distance_from_center = np.sqrt(ball_x**2 + ball_y**2)
                if distance_from_center > 0.125 or ball_z < 0.05:
                    print(f"Ball fell off after {self.step_count} control steps. Resetting...")
                    self.reset_ball(randomize=self.randomize_ball)
                    physics_step_count = 0  # Reset physics step counter
                    continue
                
                # Print status occasionally (reduced frequency when visuals are enabled)
                print_freq = self.control_freq * 10 if self.enable_visuals else self.control_freq
                if self.step_count % print_freq == 0:
                    print(f"Step: {self.step_count}, Method: {self.control_method.upper()}, "
                          f"Distance: {distance_from_center:.3f}m")
                
                self.step_count += 1
            
            # Handle keyboard input (check every physics step for responsiveness)
            keys = p.getKeyboardEvents()
            for key, state in keys.items():
                if state & p.KEY_WAS_TRIGGERED:
                    if key == ord('r'):
                        print("Resetting ball...")
                        self.reset_ball(randomize=self.randomize_ball)
                        physics_step_count = 0  # Reset physics step counter
                    elif key == ord('f'):
                        self.randomize_ball = not self.randomize_ball
                        mode = "Random" if self.randomize_ball else "Fixed"
                        print(f"Ball position mode: {mode}")
                        self.reset_ball(randomize=self.randomize_ball)
                        physics_step_count = 0  # Reset physics step counter
                    elif key == ord('p'):
                        print("Switching to PID control")
                        self.control_method = "pid"
                        self.reset_ball(randomize=self.randomize_ball)
                        physics_step_count = 0  # Reset physics step counter
                    elif key == ord('l'):
                        print("Switching to RL control")
                        if self.rl_model is not None:
                            self.control_method = "rl"
                            self.reset_ball(randomize=self.randomize_ball)
                            physics_step_count = 0  # Reset physics step counter
                        else:
                            print("RL model not available. Attempting to load...")
                            self.load_rl_model()
                            if self.rl_model is not None:
                                self.control_method = "rl"
                                self.reset_ball(randomize=self.randomize_ball)
                                physics_step_count = 0  # Reset physics step counter
                                print("✅ RL control activated!")
                            else:
                                print("❌ Still no RL model available")
                    elif key == ord('q'):
                        print("Quitting...")
                        if self.enable_visuals:
                            self.visual_thread_running = False
                        return
            
            # Always step physics at physics frequency
            p.stepSimulation()
            time.sleep(self.physics_dt)
            physics_step_count += 1


def main():
    parser = argparse.ArgumentParser(description="Ball Balancing Control Comparison")
    parser.add_argument("--control", choices=["pid", "rl"], default="pid", 
                       help="Control method to start with")
    parser.add_argument("--freq", type=int, default=50,
                       help="Control frequency in Hz (default: 50)")
    parser.add_argument("--visuals", action="store_true",
                       help="Enable visual dashboard in console (thread-safe)")
    
    args = parser.parse_args()
    
    simulator = BallBalanceComparison(control_method=args.control, control_freq=args.freq, enable_visuals=args.visuals)
    
    try:
        simulator.run_simulation()
    except KeyboardInterrupt:
        print("\nSimulation stopped by user")
    finally:
        p.disconnect()


if __name__ == "__main__":
    main()
