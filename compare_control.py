import pybullet as p
import pybullet_data
import time
import numpy as np
from pid_controller import PIDController
from lqr_controller import LQRController
import argparse
import os
import threading
import queue
import keyboard

# Optional servo control
try:
    from servo.servo_controller import ServoController
    SERVO_AVAILABLE = True
except ImportError:
    SERVO_AVAILABLE = False
    print("⚠️ Servo controller not available (missing dynamixel_sdk or servo_controller.py)")

# Optional camera integration
try:
    from camera.camera_interface import CameraSimulationInterface
    CAMERA_AVAILABLE = True
except ImportError:
    CAMERA_AVAILABLE = False
    print("⚠️ Camera interface not available (missing camera_interface.py or dependencies)")

# Optional IMU feedback
try:
    import sys
    from imu.imu_simple import SimpleBNO055Interface
    IMU_AVAILABLE = True
except ImportError:
    IMU_AVAILABLE = False
    print("⚠️ IMU interface not available (missing imu_simple.py or dependencies)")


class BallBalanceComparison:
    """
    Compare PID control vs Reinforcement Learning control
    """
    
    def __init__(self, control_method="pid", control_freq=50, enable_visuals=False, enable_servos=False, camera_mode="simulation", enable_imu=False, imu_port="COM6", imu_control=False, disable_camera_rendering=False):
        self.control_method = control_method
        self.control_freq = control_freq  # Hz - control update frequency (default 50 Hz like servos)
        self.enable_visuals = enable_visuals  # Flag to enable/disable visual indicators
        self.enable_servos = enable_servos  # Flag to enable/disable servo control
        self.enable_imu = enable_imu  # Flag to enable/disable IMU feedback
        self.imu_control = imu_control  # Flag to enable IMU control mode (table follows IMU)
        self.camera_mode = camera_mode  # Camera mode: "simulation", "hybrid", or "real"
        self.disable_camera_rendering = disable_camera_rendering  # Flag to disable camera feed display for performance
        self.setpoint_x = 0.0
        self.setpoint_y = 0.0
        self._circle_mode = False
        self._circle_angle = 0.0
        
        # TIMING PARAMETERS
        self.physics_freq = 240  # Hz - physics simulation frequency
        self.physics_dt = 1.0 / self.physics_freq  # Physics timestep
        self.control_dt = 1.0 / self.control_freq  # Control timestep
        self.physics_steps_per_control = self.physics_freq // self.control_freq  # Steps per control update

        self.control_output_limit = np.radians(10)

        # PID CONTROLLER
        self.pitch_pid = PIDController(kp=1.35, ki=0.0, kd=0.18, output_limits=(-self.control_output_limit, self.control_output_limit))
        self.roll_pid = PIDController(kp=1.35, ki=0.0, kd=0.18, output_limits=(-self.control_output_limit, self.control_output_limit))

        # LQR CONTROLLER
        self.lqr_controller = LQRController()
        
        # SERVO CONTROLLER
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
        
        # IMU controller for real-time feedback
        self.imu_interface = None
        self.imu_pitch = 0.0  # Current IMU pitch reading
        self.imu_roll = 0.0   # Current IMU roll reading
        self.imu_heading = 0.0  # Current IMU heading reading
        self.imu_connected = False
        
        # IMU offset calibration (for handling IMU baseline errors)
        self.imu_pitch_offset = 0.0  # Offset to subtract from pitch readings
        self.imu_roll_offset = 0.0   # Offset to subtract from roll readings
        self.imu_calibrated = False  # Flag indicating if offsets have been calibrated
        self.calibration_samples = []  # For collecting calibration data
        self.imu_feedback_error = (0.0, 0.0)  # Track commanded vs actual angle errors
        
        # Try to load embedded IMU calibration data first
        self.load_embedded_imu_calibration()
        
        if (self.enable_imu or self.imu_control) and IMU_AVAILABLE:
            self.imu_interface = SimpleBNO055Interface(port=imu_port)
            if self.imu_interface.connect():
                if self.imu_control:
                    print("✅ IMU control mode enabled - table will follow IMU movements")
                else:
                    print("✅ IMU feedback enabled")
                self.imu_connected = True
                # Start background thread for IMU reading
                self.imu_thread_running = True
                self.imu_thread = threading.Thread(target=self._imu_reader_thread, daemon=True)
                self.imu_thread.start()
            else:
                print("❌ Failed to connect to IMU, continuing without IMU feedback")
                self.imu_interface = None
        elif (self.enable_imu or self.imu_control):
            print("❌ IMU feedback requested but not available")
        
        # Camera interface
        self.camera_interface = None
        use_camera = camera_mode in ["hybrid", "real"] and CAMERA_AVAILABLE
        if use_camera:
            self.camera_interface = CameraSimulationInterface(use_camera=True, table_size=0.25, disable_rendering=self.disable_camera_rendering)
            print(f"✅ Camera mode: {camera_mode}")
            if self.disable_camera_rendering:
                print("📷 Camera interface initialized (rendering disabled for performance)")
            else:
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
        self.table_pitch = 0.0  # Commanded table angles
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
                from camera.camera_calibration_color import ColorCalibrationTest
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
        
        # Add simple coordinate system axes
        if self.camera_mode != "real":
            axis_length = 0.1
            p.addUserDebugLine([0, 0, 0.065], [axis_length, 0, 0.065], [1, 0, 0], lineWidth=3)  # X-axis red
            p.addUserDebugLine([0, 0, 0.065], [0, axis_length, 0.065], [0, 1, 0], lineWidth=3)  # Y-axis green  
            p.addUserDebugLine([0, 0, 0.065], [0, 0, 0.065 + axis_length], [0, 0, 1], lineWidth=3)  # Z-axis blue
            p.addUserDebugText("X", [axis_length, 0, 0.065], textColorRGB=[1, 0, 0], textSize=2)
            p.addUserDebugText("Y", [0, axis_length, 0.065], textColorRGB=[0, 1, 0], textSize=2)
            p.addUserDebugText("Z", [0, 0, 0.065 + axis_length], textColorRGB=[0, 0, 1], textSize=2)
        
        if self.camera_mode == "hybrid":
            print("🔗 Hybrid mode - camera input with simulated physics")
        
    def reset_ball(self, position=None, randomize=True):
        """Reset ball to initial position - modified for camera modes"""
        if self.camera_mode == "real":
            print("ℹ️ Real camera mode - please manually position the ball on the table")
            #input("   Press Enter when ball is positioned...")
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
                position = [-0.1, 0.00, 0.5]
                print(f"Ball reset to fixed position: ({position[0]:.2f}, {position[1]:.2f})")
        
        # 🛡️ Ensure minimum height above the table to avoid bad spawns
        if position[2] < 0.2:
            print(f"Ball Z too low ({position[2]:.2f}), adjusting to safe height.")
            position[2] = 0.3  # safe above-table height

        if hasattr(self, 'ball_id'):
            p.removeBody(self.ball_id)

        self.ball_id = p.createMultiBody(
            baseMass=0.0027,  # 
            baseCollisionShapeIndex=p.createCollisionShape(p.GEOM_SPHERE, radius=self.ball_radius),
            baseVisualShapeIndex=p.createVisualShape(
                p.GEOM_SPHERE,
                radius=self.ball_radius,
                rgbaColor=[0.9, 0.9, 0.9, 1],
                specularColor=[0.8, 0.8, 0.8]
            ),
            basePosition=position
        )

        # 🎯 CRITICAL: Apply realistic physics parameters for PLEXIGLASS table
        # Plexiglass is much more slippery than wood/metal surfaces
        p.changeDynamics(
            self.ball_id, -1,  # -1 means base link
            lateralFriction=0.15,       # Much lower friction - plexiglass is slippery!
            rollingFriction=0.005,      # Very low rolling resistance on smooth plexiglass
            spinningFriction=0.001,     # Minimal spinning friction on smooth surface
            restitution=0.4,            # Ping pong balls bounce more on hard plexiglass
            linearDamping=0.02,         # Minimal air resistance for responsiveness
            angularDamping=0.01,        # Very low rotational damping for plexiglass
            contactStiffness=3000,      # Higher stiffness for hard plexiglass surface
            contactDamping=20           # Lower contact damping for responsive motion
        )
        
        # Plexiglass table surface properties
        p.changeDynamics(
            self.table_id, -1,
            lateralFriction=0.15,       # Match ball friction - smooth plexiglass
            rollingFriction=0.005,      # Very smooth surface
            restitution=0.4             # Hard surface with some bounce
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

        # Ball target position marker
        target_marker, = ax_table.plot(
            self.setpoint_x, self.setpoint_y, 'o',
            markerfacecolor='none', markeredgecolor='green', markersize=12, markeredgewidth=2
        )
        
        # Add legend for clarity
        ax_table.legend(loc='upper right', fontsize=9)
        
        # Control effort and angles combined
        ax_control = fig.add_subplot(gs[1, :])
        ax_control.set_title('Control Actions & Table Angles', fontweight='bold')
        ax_control.set_ylim(-0.2, 0.2)
        
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

                # Update target position marker
                target_marker.set_data([data['setpoint_x']], [data['setpoint_y']])
                
                # Update combined bars (actions and angles) - only if changed significantly
                if data['imu']['connected']:
                    heights = [data['action'][0], data['action'][1], np.radians(data['imu']['pitch']), np.radians(data['imu']['roll'])]
                else:
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
                    
                    # Build status text with IMU feedback if available
                    status_lines = [
                        f"Control: {data['method'].upper():<3} | Step: {data['step']:<6} | Distance: {data['distance']:.3f}m",
                        f"Ball Pos: ({data['ball_x']:+.3f}, {data['ball_y']:+.3f}) | Velocity: {velocity_mag:.3f} m/s",
                        f"Action: [{data['action'][0]:+.3f}, {data['action'][1]:+.3f}] | Magnitude: {action_mag:.4f}",
                        f"Table: Pitch {data['pitch']:+.3f} | Roll {data['roll']:+.3f}"
                    ]

                    # Add IMU feedback if connected
                    if data['imu']['connected']:
                        imu_pitch_diff = data['imu']['pitch'] - np.degrees(data['pitch'])
                        imu_roll_diff = data['imu']['roll'] - np.degrees(data['roll'])
                        status_lines.append(
                            f"IMU: Pitch {data['imu']['pitch']:+.1f}° Roll {data['imu']['roll']:+.1f}° | "
                            f"Diff: P{imu_pitch_diff:+.1f}° R{imu_roll_diff:+.1f}°"
                        )
                    
                    status_text.set_text('\n'.join(status_lines))
                    status_text.set_color(text_color)
                
            except queue.Empty:
                pass  # No new data, keep current display
            except Exception as e:
                print(f"Dashboard update error: {e}")
            
            # Return all artists for blitting (much faster rendering)
            return [ball_marker] + list(combined_bars) + [status_text] + [target_marker]
        
        # Set up animation with optimized update rate for smoothness vs performance
        ani = FuncAnimation(fig, update_dashboard, interval=100, blit=True, cache_frame_data=False)
        
        # Show the dashboard
        plt.tight_layout()
        plt.show(block=True)  # Block to keep window open
        
        print("Dashboard window closed")
    
    def _update_visual_data(self, ball_x, ball_y, ball_vx, ball_vy, distance, control_action, step, setpoint_x, setpoint_y, imu_feedback=None):
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
                    'setpoint_x': setpoint_x,
                    'setpoint_y': setpoint_y,
                    'distance': distance,
                    'pitch': self.table_pitch,
                    'roll': self.table_roll,
                    'action': control_action if control_action else [0.0, 0.0],
                    'imu': imu_feedback if imu_feedback else {'connected': False, 'pitch': 0.0, 'roll': 0.0, 'heading': 0.0},
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
    
    def _imu_reader_thread(self):
        """Background thread to continuously read IMU data"""
        import time
        
        print("🧭 IMU reader thread started")
        
        while self.imu_thread_running and self.imu_interface:
            try:
                line = self.imu_interface.read_line()
                if line and line.startswith("DATA: "):
                    # Parse IMU data: "DATA: heading,pitch,roll"
                    try:
                        data_part = line[6:]  # Remove "DATA: " prefix
                        heading, pitch, roll = map(float, data_part.split(','))
                        
                        # Update IMU readings (thread-safe atomic operations)
                        self.imu_heading = heading
                        self.imu_pitch = pitch
                        self.imu_roll = roll
                        
                    except Exception as e:
                        # Skip malformed data
                        pass
                        
            except Exception as e:
                # If there's a connection error, try to reconnect
                print(f"⚠️ IMU reading error: {e}")
                time.sleep(0.1)  # Shorter wait before retry (was 1 second)
                
            time.sleep(0.005)  # 200Hz reading rate for IMU control (was 100Hz)
        
        print("🧭 IMU reader thread stopped")
    
    def get_imu_feedback(self):
        """Get current IMU readings as feedback for control comparison"""
        if self.imu_connected:
            # Apply offset calibration
            calibrated_pitch = self.imu_pitch - self.imu_pitch_offset
            calibrated_roll = self.imu_roll - self.imu_roll_offset
            
            return {
                'heading': self.imu_heading,
                'pitch': calibrated_pitch,
                'roll': calibrated_roll,
                'connected': True,
                'calibrated': self.imu_calibrated
            }
        else:
            return {
                'heading': 0.0,
                'pitch': 0.0,
                'roll': 0.0,
                'connected': False,
                'calibrated': False
            }
    
    def load_embedded_imu_calibration(self, filename="imu/embedded_imu_calibration.txt"):
        """Load calibration data from embedded IMU calibration file"""
        try:
            if not os.path.exists(filename):
                print(f"ℹ️ No embedded IMU calibration file found ({filename})")
                return False
            
            print(f"📁 Loading embedded IMU calibration from {filename}...")
            
            with open(filename, 'r') as f:
                lines = f.readlines()
            
            for line in lines:
                line = line.strip()
                if line.startswith('level_pitch_offset'):
                    self.imu_pitch_offset = float(line.split('=')[1].strip())
                elif line.startswith('level_roll_offset'):
                    self.imu_roll_offset = float(line.split('=')[1].strip())
            
            if self.imu_pitch_offset != 0.0 or self.imu_roll_offset != 0.0:
                self.imu_calibrated = True
                print(f"✅ Loaded embedded IMU calibration:")
                print(f"   📐 Pitch offset: {self.imu_pitch_offset:+.2f}°")
                print(f"   📐 Roll offset:  {self.imu_roll_offset:+.2f}°")
                return True
            else:
                print(f"⚠️ Calibration file found but offsets are zero")
                return False
                
        except Exception as e:
            print(f"❌ Failed to load embedded IMU calibration: {e}")
            return False
    
    def calibrate_imu_offsets(self, num_samples=50):
        """Calibrate IMU offsets by assuming table starts level"""
        import time
        
        if not self.imu_connected:
            print("❌ IMU not connected, cannot calibrate")
            return False
        
        print(f"🧭 Calibrating IMU offsets...")
        print(f"📋 Make sure your table is LEVEL and press Enter to start calibration")
        input("   Press Enter when table is level...")
        
        print(f"📊 Collecting {num_samples} samples for offset calibration...")
        self.calibration_samples = []
        
        # Collect samples
        for i in range(num_samples):
            if not self.imu_connected:
                print("❌ IMU disconnected during calibration")
                return False
            
            self.calibration_samples.append({
                'pitch': self.imu_pitch,
                'roll': self.imu_roll
            })
            
            if (i + 1) % 10 == 0:
                print(f"   Sample {i + 1}/{num_samples} - P:{self.imu_pitch:+.1f}° R:{self.imu_roll:+.1f}°")
            
            time.sleep(0.1)  # 10Hz sampling for calibration
        
        # Calculate offsets (average of all samples)
        pitch_samples = [s['pitch'] for s in self.calibration_samples]
        roll_samples = [s['roll'] for s in self.calibration_samples]
        
        self.imu_pitch_offset = np.mean(pitch_samples)
        self.imu_roll_offset = np.mean(roll_samples)
        
        # Calculate standard deviation to assess calibration quality
        pitch_std = np.std(pitch_samples)
        roll_std = np.std(roll_samples)
        
        self.imu_calibrated = True
        
        print(f"✅ IMU offset calibration complete!")
        print(f"   📐 Pitch offset: {self.imu_pitch_offset:+.2f}° (std: {pitch_std:.2f}°)")
        print(f"   📐 Roll offset:  {self.imu_roll_offset:+.2f}° (std: {roll_std:.2f}°)")
        
        if pitch_std > 0.5 or roll_std > 0.5:
            print(f"⚠️ High standard deviation detected - table may not be stable")
            print(f"   Consider recalibrating on a more stable surface")
        else:
            print(f"✅ Calibration quality: Good (low noise)")
        
        return True
    
    def get_calibrated_imu_angles(self):
        """Get IMU angles with offset compensation applied"""
        if not self.imu_connected:
            return 0.0, 0.0
        
        # Apply offsets to get true angles relative to calibrated level
        pitch_rad = np.radians(self.imu_pitch - self.imu_pitch_offset)
        roll_rad = np.radians(self.imu_roll - self.imu_roll_offset)
        
        return pitch_rad, roll_rad
    
    
    def load_rl_model(self):
        """Load trained RL model"""
        try:
            from stable_baselines3 import SAC
            
            # Try multiple model paths
            model_paths = [
                #"models/best_model",  # Check main models folder first (for backward compatibility)
                "reinforcement_learning/models/best_model"  # New location
                #"reinforcement_learning/SAC_models/best_model"  # Alternative path
            ]
            
            for model_path in model_paths:
                full_path = model_path + ".zip"
                print(f"Checking for model at: {full_path}")
                if os.path.exists(full_path):
                    try:
                        self.rl_model = SAC.load(model_path)
                        print(f"✅ RL model loaded successfully from {model_path}")
                        return
                    except Exception as e:
                        print(f"❌ Error loading model from {model_path}: {e}")
                        continue
                else:
                    print(f"❌ Model file not found: {full_path}")
            
            # If we get here, no model was loaded
            print("❌ No RL models found. Checked locations:")
            for path in ["models/", "reinforcement_learning/models/"]:
                if os.path.exists(path):
                    print(f"   📁 {path}:")
                    for file in os.listdir(path):
                        print(f"      - {file}")
                else:
                    print(f"   📁 {path}: directory doesn't exist")
            
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
    
    def set_setpoint(self, x, y):
        self.setpoint_x = x
        self.setpoint_y = y
        if hasattr(self, 'camera_interface') and self.camera_interface:
            self.camera_interface.set_target_position(x, y)  # Update camera interface if applicable
        
        print(f"Setpoint updated: ({x:.3f}, {y:.3f})")

        # Immediately update dashboard with new setpoint (if visuals enabled)
        if hasattr(self, 'enable_visuals') and self.enable_visuals:
            try:
                observation = self.get_observation()
                ball_x, ball_y, ball_vx, ball_vy = observation[0], observation[1], observation[2], observation[3]
                distance = np.sqrt((ball_x - self.setpoint_x)**2 + (ball_y - self.setpoint_y)**2)
                control_action = [self.table_pitch, self.table_roll]
                step = getattr(self, 'step_count', 0)
                imu_feedback = self.get_imu_feedback() if hasattr(self, 'get_imu_feedback') else None
                self._update_visual_data(
                    ball_x, ball_y, ball_vx, ball_vy, distance, control_action, step,
                    self.setpoint_x, self.setpoint_y, imu_feedback
                )
            except Exception as e:
                print(f"[set_setpoint] Visual update failed: {e}")
    
    def pid_control(self, ball_x, ball_y, ball_vx, ball_vy):
        """PD control logic (no integral term needed for ball balancing)"""
        # Use control timestep for PID updates
        
        # Remove dead zone for maximum responsiveness - every movement should be corrected
        ball_x_corrected = ball_x
        ball_y_corrected = ball_y

        ball_x_corrected = ball_x - self.setpoint_x
        ball_y_corrected = ball_y - self.setpoint_y
        
        if self.camera_mode in ["hybrid", "real"]:
            # In hybrid and real modes, coordinate system is swapped to match camera
            pitch_angle = -self.pitch_pid.update(ball_x_corrected, self.control_dt)  # ball_x controls pitch
            roll_angle = self.roll_pid.update(ball_y_corrected, self.control_dt)     # ball_y controls roll
        else:
            # In simulation mode, use original mapping
            pitch_angle = -self.pitch_pid.update(ball_y_corrected, self.control_dt)  # ball_y controls pitch
            roll_angle = self.roll_pid.update(ball_x_corrected, self.control_dt)     # ball_x controls roll


        return pitch_angle, roll_angle
    
    def rl_control(self, observation):
        """RL control logic"""
        if self.rl_model is None:
            return 0.0, 0.0
        
        action, _ = self.rl_model.predict(observation, deterministic=True)
        # Only print RL actions occasionally to avoid spam
        # if self.step_count % (self.control_freq * 2) == 0:  # Print every 2 seconds
        print(f"RL action: {action}")
        if self.camera_mode in ["hybrid", "real"]:
            action = [-action[1], -action[0]]
        else:
            action = action[0], action[1]  # Reverse order for simulation
        return action  # pitch_change, roll_change
    
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
        if self.imu_connected:
            print("  'c' - Calibrate IMU offsets (make table level first!)")
        print("  'q' - Quit")
        
        if self.imu_control:
            print(f"🧭 IMU CONTROL MODE - Table follows IMU movements")
            print(f"   📋 Calibration status: {'✅ Calibrated (precise)' if self.imu_calibrated else '⚠️ Uncalibrated (direct raw angles)'}")
            print(f"   💡 Press 'c' to calibrate for precise zero-reference control")
            print(f"   🎮 Table responds immediately to IMU movements!")
        else:
            print(f"Ball position mode: {'Random' if self.randomize_ball else 'Fixed'}")
        
        physics_step_count = 0  # Track physics steps

        # Keyboard callback to toggle circle mode (e.g., on 'c' key release)
        def on_i_release(e):
            self._circle_mode = not self._circle_mode  # Toggle on/off

        

        def on_w_release(e):
            self.setpoint_y += 0.05
            self.set_setpoint(self.setpoint_x, self.setpoint_y)

        def on_s_release(e):
            self.setpoint_y -= 0.05
            self.set_setpoint(self.setpoint_x, self.setpoint_y)

        def on_a_release(e):
            self.setpoint_x -= 0.05
            self.set_setpoint(self.setpoint_x, self.setpoint_y)

        def on_d_release(e):
            self.setpoint_x += 0.05
            self.set_setpoint(self.setpoint_x, self.setpoint_y)

        def on_1_release(e):
            self.set_setpoint(-0.07, 0.07)

        def on_2_release(e):
            self.set_setpoint(0.07, 0.07)

        def on_3_release(e):
            self.set_setpoint(0.07, -0.07)

        def on_4_release(e):
            self.set_setpoint(-0.07, -0.07)

        def on_0_release(e):
            self.set_setpoint(0.0, 0.0)

        keyboard.on_release_key('1', on_1_release)
        keyboard.on_release_key('2', on_2_release)
        keyboard.on_release_key('3', on_3_release)
        keyboard.on_release_key('4', on_4_release)
        keyboard.on_release_key('0', on_0_release)

        keyboard.on_release_key('i', on_i_release)
        keyboard.on_release_key('w', on_w_release)
        keyboard.on_release_key('s', on_s_release)
        keyboard.on_release_key('a', on_a_release)
        keyboard.on_release_key('d', on_d_release)
        
        while True:
            # Only run control logic at the specified control frequency
            if physics_step_count % self.physics_steps_per_control == 0:
                # Get observation (includes position + estimated velocity)
                observation = self.get_observation()

                ball_x, ball_y, ball_vx, ball_vy = observation[0], observation[1], observation[2], observation[3]

                # Get ball height for collision detection
                if self.camera_mode == "real":
                    # In real camera mode, assume ball is on the table
                    ball_z = 0.08  # Reasonable height for ball on table
                else:
                    ball_pos, _ = p.getBasePositionAndOrientation(self.ball_id)
                    ball_z = ball_pos[2]
                    
                    if self.camera_mode == "hybrid":
                        # In hybrid mode, update simulation ball position to match camera
                        p.resetBasePositionAndOrientation(self.ball_id, [ball_x, ball_y, ball_z], [0, 0, 0, 1])
                
                # Control logic - runs at control frequency
                control_action = None
                if self.imu_control:
                    # IMU Control Mode: Table follows IMU movements
                    if self.imu_connected:
                        if self.imu_calibrated:
                            # Use calibrated angles - these represent the TRUE physical table angles
                            pitch_rad, roll_rad = self.get_calibrated_imu_angles()
                        else:
                            # Use raw angles for immediate response (no calibration required)
                            pitch_rad = np.radians(self.imu_pitch)
                            roll_rad = np.radians(self.imu_roll)
                        
                        # Set simulation table to match the TRUE physical table angles
                        self.table_pitch = pitch_rad
                        self.table_roll = roll_rad
                        
                        # Limit to safe angles
                        self.table_pitch = np.clip(self.table_pitch, -0.15, 0.15)  # ~8.6 degrees max
                        self.table_roll = np.clip(self.table_roll, -0.15, 0.15)
                        
                        control_action = [self.table_pitch, self.table_roll]
                    else:
                        # If IMU not connected, keep table level
                        self.table_pitch = 0.0
                        self.table_roll = 0.0
                        control_action = [0.0, 0.0]
                elif self.control_method == "lqr":
                    pitch_angle, roll_angle = self.lqr_controller.control(ball_x, ball_y, ball_vx, ball_vy)
                    # For LQR, these are absolute angles
                    self.table_pitch = pitch_angle
                    self.table_roll = roll_angle
                    control_action = [pitch_angle, roll_angle]
                elif self.control_method == "pid":
                    pitch_angle, roll_angle = self.pid_control(ball_x, ball_y, ball_vx, ball_vy)
                    
                    # IMU Feedback Correction (if IMU available and not in IMU control mode)
                    # Re-enabled with conservative gains to help with edge recovery
                    if self.imu_connected and self.imu_calibrated:
                        # Get actual table angles from IMU
                        actual_pitch, actual_roll = self.get_calibrated_imu_angles()
                        
                        
                        # Calculate angle errors (commanded vs actual)
                        pitch_error = pitch_angle - actual_pitch
                        roll_error = roll_angle - actual_roll
                    
                        
                        # Apply feedback correction to reduce the error (subtract error, don't add it!)
                        # More responsive gain to complement PID control and speed up convergence
                        pitch_angle -= 0.125 * pitch_error  # More responsive correction (was 0.05)
                        roll_angle -= 0.125 * roll_error
                
                        
                        # Store feedback info for display
                        self.imu_feedback_error = (pitch_error, roll_error)
                    
                    # Ensure final angles stay within servo limits after IMU correction
                    pitch_angle = np.clip(pitch_angle, -self.control_output_limit, self.control_output_limit)  # ±3.2°
                    roll_angle = np.clip(roll_angle, -self.control_output_limit, self.control_output_limit)   # ±3.2°
                
                    
                    # For PID, these are absolute angles
                    self.table_pitch = pitch_angle
                    self.table_roll = roll_angle
                    control_action = [pitch_angle, roll_angle]
                else:  # RL
                    pitch_change, roll_change = self.rl_control(observation)
                    
                    # Apply RL action to get new intended table angles
                    new_table_pitch = self.table_pitch + pitch_change
                    new_table_roll = self.table_roll + roll_change
                    
                    # IMU Feedback Correction for RL (if IMU available and not in IMU control mode)
                    if self.imu_connected and self.imu_calibrated:
                        # Get actual current table angles from IMU
                        actual_pitch, actual_roll = self.get_calibrated_imu_angles()
                        
                        # Calculate error between current simulation angle and actual IMU angle
                        sim_imu_pitch_error = self.table_pitch - actual_pitch
                        sim_imu_roll_error = self.table_roll - actual_roll
                        
                        # Correct the new target angles based on the simulation vs reality offset
                        # This compensates for the fact that simulation might be out of sync with reality
                        corrected_pitch = new_table_pitch - 0.1 * sim_imu_pitch_error
                        corrected_roll = new_table_roll - 0.1 * sim_imu_roll_error
                        
                        # Store feedback info for display (showing sim vs IMU error)
                        self.imu_feedback_error = (sim_imu_pitch_error, sim_imu_roll_error)
                        
                        # Use corrected angles
                        self.table_pitch = corrected_pitch
                        self.table_roll = corrected_roll

                        self.table_pitch = new_table_pitch
                        self.table_roll = new_table_roll
                    else:
                        # No IMU feedback - use RL action directly
                        self.table_pitch = new_table_pitch
                        self.table_roll = new_table_roll
                    # Clip to reasonable limits
                    self.table_pitch = np.clip(self.table_pitch, -0.1, 0.1)
                    self.table_roll = np.clip(self.table_roll, -0.1, 0.1)
                    control_action = [pitch_change, roll_change]
                
                if self.camera_mode == "real":
                    # In real camera mode, no PyBullet simulation to update
                    pass
                elif self.camera_mode == "hybrid":
                    # In hybrid mode, match real-world right-handed coordinate system
                    quat = p.getQuaternionFromEuler([self.table_roll, self.table_pitch, 0])
                    p.resetBasePositionAndOrientation(self.table_id, self.table_start_pos, quat)
                else:
                    # In simulation mode, use original PyBullet convention for PID compatibility
                    quat = p.getQuaternionFromEuler([self.table_pitch, self.table_roll, 0])
                    p.resetBasePositionAndOrientation(self.table_id, self.table_start_pos, quat)
                
                # Send COMMANDED angles to servo controller (real hardware gets commands)
                if self.servo_controller:
                    self.servo_controller.set_table_angles(self.table_pitch, self.table_roll)
                
                # Get IMU feedback for comparison/monitoring
                imu_feedback = self.get_imu_feedback()
                
                # Update visual data (thread-safe) - every 10 steps for smoother updates
                if self.step_count % 10 == 0:
                    distance_from_center = np.sqrt(ball_x**2 + ball_y**2)
                    self._update_visual_data(ball_x, ball_y, ball_vx, ball_vy, distance_from_center, control_action, self.step_count, self.setpoint_x, self.setpoint_y, imu_feedback)
                
                # Check if ball fell off - 25cm table (radius = 0.125m)
                # Check if ball fell off - 25cm table (radius = 0.125m)
                distance_from_center = np.sqrt(ball_x**2 + ball_y**2)
                ball_fell = distance_from_center > 0.125 and ball_z < 0.5

                if self._circle_mode:
                    # 2π radians per 5 seconds at 60Hz: increment = 2π / (60*5)
                    self._circle_angle += 2 * np.pi / 60
                    radius = 0.03  # or whatever radius you want
                    self.setpoint_x = radius * np.cos(self._circle_angle)
                    self.setpoint_y = radius * np.sin(self._circle_angle)
                    self.set_setpoint(self.setpoint_x, self.setpoint_y)

                if ball_fell and not self._ball_reset:
                    print(f"Ball fell off after {self.step_count} control steps. Resetting...")

                    # Reset servos to level position when ball falls off
                    if self.servo_controller:
                        print("Resetting servos to level position...")
                        self.servo_controller.set_table_angles(0.0, 0.0)
                    
                    # Reset PID controllers and table angles
                    self.pitch_pid.reset()
                    self.roll_pid.reset()
                    self.table_pitch = 0.0
                    self.table_roll = 0.0

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
                    if self.imu_control:
                        # IMU control mode status
                        if self.imu_connected:
                            imu_feedback = self.get_imu_feedback()
                            status_line = f"Step: {self.step_count}, IMU CONTROL, Distance: {distance_from_center:.3f}m"
                            status_line += f" | IMU: P{self.imu_pitch:+.1f}° R{self.imu_roll:+.1f}°"
                            status_line += f" | Table: P{np.degrees(self.table_pitch):+.1f}° R{np.degrees(self.table_roll):+.1f}°"
                            if self.imu_calibrated:
                                status_line += " | 📐 CALIBRATED"
                            else:
                                status_line += " | 🔄 RAW"
                        else:
                            status_line = f"Step: {self.step_count}, IMU CONTROL - ❌ IMU NOT CONNECTED"
                    else:
                        # Normal PID/RL control mode status
                        status_line = f"Step: {self.step_count}, Method: {self.control_method.upper()}, Distance: {distance_from_center:.3f}m"
                        
                        # Add IMU feedback to status if connected
                        if imu_feedback['connected']:
                            status_line += f" | IMU: P{imu_feedback['pitch']:+.1f}° R{imu_feedback['roll']:+.1f}°"
                            
                            # Show feedback errors if using IMU feedback correction
                            if hasattr(self, 'imu_feedback_error') and (abs(self.imu_feedback_error[0]) > 0.1 or abs(self.imu_feedback_error[1]) > 0.1):
                                pitch_err, roll_err = self.imu_feedback_error
                                status_line += f" | Err: P{np.degrees(pitch_err):+.1f}° R{np.degrees(roll_err):+.1f}° | 🔄 FB"
                    
                    print(status_line)
                
                self.step_count += 1
            
            # Handle keyboard input (check every physics step for responsiveness)
            if self.camera_mode != "real":
                # Only use PyBullet keyboard events if PyBullet is running
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
                        elif key == ord('c'):
                            # Calibrate IMU offsets
                            if self.imu_connected:
                                print("🧭 Starting IMU calibration...")
                                if self.calibrate_imu_offsets():
                                    print("✅ IMU calibration completed!")
                                else:
                                    print("❌ IMU calibration failed!")
                            else:
                                print("❌ IMU not connected - cannot calibrate")
                        elif key == ord('q'):
                            print("Quitting...")
                            if self.enable_visuals:
                                self.visual_thread_running = False
                            if self.enable_imu:
                                self.imu_thread_running = False
                            if self.servo_controller:
                                self.servo_controller.disconnect()
                            if self.camera_interface:
                                self.camera_interface.stop_tracking()
                                self.camera_interface.cleanup()
                            if self.imu_interface:
                                self.imu_interface.cleanup()
                            return
                       
            else:
                # In real mode, handle keyboard input differently or use a simple loop check
                # For now, just provide terminal-based control
                if keyboard.is_pressed('q'):
                    print("Quitting...")
                    break
            
            # Always step physics at physics frequency (only if PyBullet is running)
            if self.camera_mode != "real":
                p.stepSimulation()
                time.sleep(self.physics_dt)
            else:
                # In real mode, just sleep at control frequency
                time.sleep(self.control_dt)
            physics_step_count += 1
    


def main():
    parser = argparse.ArgumentParser(description="Ball Balancing Control Comparison")
    parser.add_argument("--control", choices=["pid", "rl","lqr"], default="pid", 
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
    parser.add_argument("--imu", action="store_true",
                       help="Enable IMU feedback for real-time angle monitoring")
    parser.add_argument("--imu-control", action="store_true",
                       help="Enable IMU control mode (table follows IMU movements)")
    parser.add_argument("--imu-port", default="COM6",
                       help="COM port for IMU connection (default: COM6)")
    parser.add_argument("--check-imu", action="store_true",
                       help="Quick check of IMU calibration accuracy (no simulation)")
    parser.add_argument("--disable-camera-rendering", action="store_true",
                       help="Disable camera feed display for better performance (keeps ball detection active)")
    args = parser.parse_args()
    
    # Quick IMU calibration check mode
    if args.check_imu:
        print("🔍 Quick IMU Calibration Check")
        print("=" * 40)
        
        try:
            # Import the calibration checker
            sys.path.append('imu')
            from imu.check_calibration import CalibrationChecker
            
            checker = CalibrationChecker(args.imu_port)
            checker.run_full_check()
            
        except ImportError:
            print("❌ Calibration checker not available")
            print("   Make sure check_calibration.py is in the imu/ folder")
        except Exception as e:
            print(f"❌ Error during calibration check: {e}")
        
        return  # Exit after check
    
    simulator = BallBalanceComparison(
        control_method=args.control, 
        control_freq=args.freq, 
        enable_visuals=args.visuals,
        enable_servos=args.servos,
        camera_mode=args.camera,
        enable_imu=args.imu,
        imu_port=args.imu_port,
        disable_camera_rendering=args.disable_camera_rendering,
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
        
        # Additional instructions for IMU mode
        if args.imu:
            print("\n🧭 IMU Feedback Mode Instructions:")
            print("   1. Ensure BNO055 IMU is connected via Arduino")
            print("   2. IMU will provide real-time table angle feedback")
            print("   3. Useful for comparing simulation vs real hardware")
            print("   4. Dashboard will show simulation vs IMU angle differences")
            print(f"   5. Using COM port: {args.imu_port}")
            input("\n   Press Enter to continue...")
        
        # Additional instructions for IMU control mode
        if args.imu_control:
            print("\n🎮 IMU Control Mode Instructions:")
            print("   1. Ensure BNO055 IMU is connected via Arduino")
            print("   2. Table will respond IMMEDIATELY to IMU movements")
            print("   3. Calibration is OPTIONAL - press 'c' for zero-reference precision")
            print("   4. Physically tilt your table - simulation follows instantly!")
            print("   5. Great for testing IMU responsiveness and ball physics")
            print(f"   6. Using COM port: {args.imu_port}")
            print("\n   📋 This mode disables PID/RL control - table follows your IMU directly")
            print("   🚀 No calibration required - works immediately!")
            input("\n   Press Enter to continue...")
        
        simulator.run_simulation()
    except KeyboardInterrupt:
        print("\nSimulation stopped by user")
        if hasattr(simulator, "servo_controller") and simulator.servo_controller:
            print("Returning table to level (0, 0)...")
            simulator.servo_controller.set_table_angles(0.0, 0.0)
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Only disconnect if PyBullet was connected
        try:
            if args.camera != "real":
                p.disconnect()
        except:
            pass  # Ignore disconnect errors


if __name__ == "__main__":
    main()
