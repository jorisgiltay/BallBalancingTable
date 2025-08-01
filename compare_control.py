import pybullet as p
import pybullet_data
import time
import numpy as np
from pid_controller import PIDController
import argparse
import os
import threading
import queue

# Optional servo control
try:
    from servo_controller import ServoController
    SERVO_AVAILABLE = True
except ImportError:
    SERVO_AVAILABLE = False
    print("⚠️ Servo controller not available (missing dynamixel_sdk or servo_controller.py)")

# Optional camera integration
try:
    from camera_interface import CameraSimulationInterface
    CAMERA_AVAILABLE = True
except ImportError:
    CAMERA_AVAILABLE = False
    print("⚠️ Camera interface not available (missing camera_interface.py or dependencies)")


class BallBalanceComparison:
    """
    Compare PID control vs Reinforcement Learning control
    """
    
    def __init__(self, control_method="pid", control_freq=50, enable_visuals=False, enable_servos=False, camera_mode="simulation"):
        self.control_method = control_method
        self.control_freq = control_freq  # Hz - control update frequency (default 50 Hz like servos)
        self.enable_visuals = enable_visuals  # Flag to enable/disable visual indicators
        self.enable_servos = enable_servos  # Flag to enable/disable servo control
        self.camera_mode = camera_mode  # Camera mode: "simulation", "hybrid", or "real"
        
        # Timing parameters
        self.physics_freq = 240  # Hz - physics simulation frequency
        self.physics_dt = 1.0 / self.physics_freq  # Physics timestep
        self.control_dt = 1.0 / self.control_freq  # Control timestep
        self.physics_steps_per_control = self.physics_freq // self.control_freq  # Steps per control update
        
        # PID controllers (create these BEFORE setup_simulation)
        self.pitch_pid = PIDController(kp=10.0, ki=0.1, kd=2.0, output_limits=(-0.05, 0.05))
        self.roll_pid = PIDController(kp=10.0, ki=0.1, kd=2.0, output_limits=(-0.05, 0.05))
        
        # Servo controller
        self.servo_controller = None
        if self.enable_servos and SERVO_AVAILABLE:
            self.servo_controller = ServoController()
            if self.servo_controller.connect():
                print("✅ Servo control enabled")
            else:
                print("❌ Failed to connect to servos, continuing without servo control")
                self.servo_controller = None
        elif self.enable_servos:
            print("❌ Servo control requested but not available")
        
        # Camera interface
        self.camera_interface = None
        use_camera = camera_mode in ["hybrid", "real"] and CAMERA_AVAILABLE
        if use_camera:
            self.camera_interface = CameraSimulationInterface(use_camera=True, table_size=0.25)
            print(f"✅ Camera mode: {camera_mode}")
            print("📷 Camera interface initialized")
            if camera_mode == "real":
                print("⚠️ Real camera mode - make sure PyBullet simulation represents actual table")
        elif camera_mode != "simulation":
            print(f"❌ Camera mode '{camera_mode}' requested but camera interface not available")
            self.camera_mode = "simulation"
        
        # Initialize ball reset tracking
        self._ball_reset = False
        
        # Thread-safe visual system with optimized queue
        if self.enable_visuals:
            self.visual_queue = queue.Queue(maxsize=5)  # Small buffer to keep data fresh
            self.visual_thread_running = True
            self.visual_thread = threading.Thread(target=self._visual_matplotlib_thread, daemon=True)
        
        self.setup_simulation()
        
        # Start visual thread if enabled
        if self.enable_visuals:
            self.visual_thread.start()
        
        # Start camera tracking if using camera
        if self.camera_mode in ["hybrid", "real"] and self.camera_interface:
            self.camera_interface.start_tracking()
            print("📷 Camera tracking started")
        
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
    
    def calibrate_camera(self):
        """Calibrate camera for table detection"""
        if self.camera_mode == "simulation":
            print("ℹ️ Camera calibration not needed in simulation mode")
            return True
        
        print("🎯 Starting camera calibration...")
        print("📋 Options:")
        print("   1. Interactive calibration (recommended)")
        print("   2. Run external script (may have Unicode issues)")
        print("   3. Skip calibration (use existing data)")
        
        choice = input("\nChoose option (1-3): ").strip()
        
        if choice == "3":
            print("ℹ️ Skipping calibration, using existing calibration data")
            if self.camera_interface:
                success = self.camera_interface.load_existing_calibration()
                return success
            return False
        elif choice == "2":
            return self._run_external_calibration()
        else:  # Default to option 1
            return self._run_interactive_calibration()
    
    def _run_external_calibration(self):
        """Run the external calibration script"""
        print("📋 Running external camera_calibration_color.py script...")
        print("   Make sure:")
        print("   - RealSense camera is connected")
        print("   - 35x35cm base plate with 4 blue markers is visible")
        print("   - Table is well-lit and no ball is present")
        
        import subprocess
        import sys
        
        try:
            # Set environment to handle Unicode properly
            env = os.environ.copy()
            env['PYTHONIOENCODING'] = 'utf-8'
            
            # Run the camera calibration script with proper encoding
            result = subprocess.run([sys.executable, "camera_calibration_color.py"], 
                                  capture_output=True, text=True, timeout=300,
                                  env=env, encoding='utf-8', errors='replace')
            
            if result.returncode == 0:
                print("✅ Camera calibration completed successfully")
                print("📁 Calibration data saved to calibration_data/ folder")
                
                # Reload calibration data in the camera interface
                if self.camera_interface:
                    success = self.camera_interface.load_existing_calibration()
                    if success:
                        print("✅ Calibration data loaded into camera interface")
                        return True
                    else:
                        print("⚠️ Calibration completed but failed to load into camera interface")
                        return False
                return True
            else:
                print("❌ Camera calibration failed")
                if result.stderr:
                    print(f"Error output: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            print("❌ Camera calibration timed out (5 minutes)")
            return False
        except FileNotFoundError:
            print("❌ camera_calibration_color.py script not found")
            print("   Make sure you're running from the project root directory")
            return False
        except Exception as e:
            print(f"❌ Error running calibration: {e}")
            return False
    
    def _run_interactive_calibration(self):
        """Run a simplified interactive calibration"""
        print("🎯 Interactive Camera Calibration")
        print("=" * 40)
        print("This will guide you through calibrating the camera manually.")
        print("")
        print("Setup requirements:")
        print("- 35x35cm wooden base plate")
        print("- 4 blue markers (4x4cm) at the corners") 
        print("- RealSense camera positioned above")
        print("- Good lighting, no ball on table")
        print("")
        
        # Ask user if they want to proceed
        proceed = input("Ready to start calibration? (y/n): ").strip().lower()
        if proceed != 'y':
            print("❌ Calibration cancelled")
            return False
        
        try:
            # Import and run the calibration directly 
            import sys
            sys.path.append('.')
            
            # Try to import the calibration module
            try:
                from camera_calibration_color import ColorCalibrationTest
            except ImportError as e:
                print(f"❌ Could not import calibration module: {e}")
                return False
            
            # Create and run calibration
            calibrator = ColorCalibrationTest()
            
            if not calibrator.initialize_camera():
                print("❌ Failed to initialize camera for calibration")
                return False
            
            print("📸 Taking 10 calibration samples...")
            print("Keep the setup steady during calibration...")
            
            success = calibrator.run_calibration(num_samples=10)
            calibrator.cleanup()
            
            if success:
                print("✅ Interactive calibration completed successfully!")
                
                # Reload calibration data
                if self.camera_interface:
                    reload_success = self.camera_interface.load_existing_calibration()
                    if reload_success:
                        print("✅ New calibration data loaded")
                        return True
                    else:
                        print("⚠️ Calibration saved but failed to reload")
                        return False
                return True
            else:
                print("❌ Interactive calibration failed")
                return False
                
        except KeyboardInterrupt:
            print("\n🛑 Calibration cancelled by user")
            return False
        except Exception as e:
            print(f"❌ Error during interactive calibration: {e}")
            import traceback
            traceback.print_exc()
            return False
        
    def setup_simulation(self):
        """Initialize PyBullet simulation - modified for camera modes"""
        if self.camera_mode == "real":
            # In real camera mode, don't create PyBullet simulation
            print("ℹ️ Real camera mode - skipping PyBullet simulation setup")
            return
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
        
        if self.camera_mode == "hybrid":
            print("🔗 Hybrid mode - camera input with simulated physics")
        
    def reset_ball(self, position=None, randomize=True):
        """Reset ball to initial position - modified for camera modes"""
        if self.camera_mode == "real":
            print("ℹ️ Real camera mode - please manually position the ball on the table")
            input("   Press Enter when ball is positioned...")
            return

        if position is None:
            if randomize:
                # Random position like in RL training - adjusted for 25cm table
                position = [
                    np.random.uniform(-0.11, 0.11),
                    np.random.uniform(-0.11, 0.11),
                    0.5
                ]
                print(f"Ball reset to random position: ({position[0]:.2f}, {position[1]:.2f})")
            else:
                position = [0.06, 0.08, 0.5]
                print(f"Ball reset to fixed position: ({position[0]:.2f}, {position[1]:.2f})")
        
        # 🛡️ Ensure minimum height above the table to avoid bad spawns
        if position[2] < 0.2:
            print(f"Ball Z too low ({position[2]:.2f}), adjusting to safe height.")
            position[2] = 0.5  # safe above-table height

        if hasattr(self, 'ball_id'):
            p.removeBody(self.ball_id)

        self.ball_id = p.createMultiBody(
            baseMass=0.0027,
            baseCollisionShapeIndex=p.createCollisionShape(p.GEOM_SPHERE, radius=self.ball_radius),
            baseVisualShapeIndex=p.createVisualShape(
                p.GEOM_SPHERE,
                radius=self.ball_radius,
                rgbaColor=[0.9, 0.9, 0.9, 1],
                specularColor=[0.8, 0.8, 0.8]
            ),
            basePosition=position
        )

        # Reset system
        self.pitch_pid.reset()
        self.roll_pid.reset()
        self.table_pitch = 0.0
        self.table_roll = 0.0
        self.prev_ball_pos = None
        self.step_count = 0

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
        """Get current state observation - now with camera support"""
        if self.camera_mode == "simulation":
            # Use original simulation-based observation
            return self._get_simulation_observation()
        
        # Get ball state from camera interface
        ball_x, ball_y, ball_vx, ball_vy = self.camera_interface.get_ball_state(
            simulation_ball_id=self.ball_id if hasattr(self, 'ball_id') else None,
            pybullet_module=p if self.camera_mode == "hybrid" else None
        )
        
        # Update previous position for next velocity calculation (maintain compatibility)
        self.prev_ball_pos = [ball_x, ball_y]
        
        # Return observation in same format as original
        observation = np.array([
            ball_x, ball_y, ball_vx, ball_vy, self.table_pitch, self.table_roll
        ], dtype=np.float32)
        
        return observation
    
    def _get_simulation_observation(self):
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
                
                if getattr(self, "camera_mode", None) == "hybrid":
                    # In hybrid mode, update simulation ball position to match camera
                    p.resetBasePositionAndOrientation(self.ball_id, [ball_x, ball_y, ball_z], [0, 0, 0, 1])
                
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
                
                # Send angles to servo controller if enabled
                if self.servo_controller:
                    self.servo_controller.set_table_angles(self.table_pitch, self.table_roll)
                
                # Update visual data (thread-safe) - every 10 steps for smoother updates
                if self.step_count % 10 == 0:
                    distance_from_center = np.sqrt(ball_x**2 + ball_y**2)
                    self._update_visual_data(ball_x, ball_y, ball_vx, ball_vy, distance_from_center, control_action, self.step_count)
                
                # Check if ball fell off - 25cm table (radius = 0.125m)
                # Check if ball fell off - 25cm table (radius = 0.125m)
                distance_from_center = np.sqrt(ball_x**2 + ball_y**2)
                ball_fell = distance_from_center > 0.125 and ball_z < 0.5

                if ball_fell and not self._ball_reset:
                    print(f"Ball fell off after {self.step_count} control steps. Resetting...")

                    if getattr(self, "camera_mode", None) == "hybrid":
                        print("Press R to reset the ball")
                    else:
                        self.reset_ball(randomize=self.randomize_ball)
                        
                    self._ball_reset = True  # Prevent further resets
                    physics_step_count = 0  # Reset physics step counter
                    continue

                # If ball is back on the table (indicating it's been successfully reset or placed)
                elif not ball_fell:
                    self._ball_reset = False  # Allow future resets

                
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
                        if self.servo_controller:
                            self.servo_controller.disconnect()
                        if self.camera_interface:
                            self.camera_interface.stop_tracking()
                            self.camera_interface.cleanup()
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
    parser.add_argument("--servos", action="store_true",
                       help="Enable servo control for real hardware")
    parser.add_argument("--camera", choices=["simulation", "hybrid", "real"], default="simulation",
                       help="Camera mode: simulation (no camera), hybrid (camera + physics), real (camera only)")
    parser.add_argument("--calibrate", action="store_true",
                       help="Perform camera calibration before starting")
    
    args = parser.parse_args()
    
    simulator = BallBalanceComparison(
        control_method=args.control, 
        control_freq=args.freq, 
        enable_visuals=args.visuals,
        enable_servos=args.servos,
        camera_mode=args.camera
    )
    
    try:
        # Calibration step
        if args.calibrate:
            if not simulator.calibrate_camera():
                print("❌ Calibration failed, exiting")
                return
        
        # Additional instructions for camera modes
        if args.camera == "real":
            print("\n📋 Real Camera Mode Instructions:")
            print("   1. Ensure RealSense camera is connected and positioned above table")
            print("   2. Table should be well-lit with good contrast")
            print("   3. Use white ping pong ball for best detection")
            print("   4. Control outputs will be connected to servos" + (" (enabled)" if args.servos else " (disabled)"))
            input("\n   Press Enter to continue...")
        elif args.camera == "hybrid":
            print("\n📋 Hybrid Mode Instructions:")
            print("   1. RealSense camera provides ball position")
            print("   2. PyBullet simulates physics and visualizes control")
            print("   3. Great for testing camera integration before hardware deployment")
            print("   4. Servo control" + (" enabled" if args.servos else " disabled"))
            input("\n   Press Enter to continue...")
        
        simulator.run_simulation()
    except KeyboardInterrupt:
        print("\nSimulation stopped by user")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        p.disconnect()


if __name__ == "__main__":
    main()
