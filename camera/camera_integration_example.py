"""
Camera Integration Example for Ball Balancing Table

This script demonstrates how to integrate the RealSense camera
with the existing ball balancing control system.

Usage:
1. For simulation mode (default):
   python camera_integration_example.py --mode simulation

2. For real camera mode:
   python camera_integration_example.py --mode camera --calibrate

3. For hybrid mode (camera position, simulated physics):
   python camera_integration_example.py --mode hybrid
"""

import argparse
import time
import numpy as np
from camera.camera_interface import CameraSimulationInterface
from pid_controller import PIDController

# Import PyBullet for simulation compatibility
try:
    import pybullet as p
    import pybullet_data
    PYBULLET_AVAILABLE = True
except ImportError:
    PYBULLET_AVAILABLE = False
    print("⚠️ PyBullet not available - camera-only mode")


class BallBalanceWithCamera:
    """
    Ball balancing system with camera integration
    """
    
    def __init__(self, mode: str = "simulation", table_size: float = 0.25):
        """
        Initialize the system
        
        Args:
            mode: "simulation", "camera", or "hybrid"
            table_size: Table size in meters
        """
        self.mode = mode
        self.table_size = table_size
        
        # Control system
        self.pitch_pid = PIDController(kp=10.0, ki=0.1, kd=2.0, output_limits=(-0.05, 0.05))
        self.roll_pid = PIDController(kp=10.0, ki=0.1, kd=2.0, output_limits=(-0.05, 0.05))
        
        # Camera interface
        use_camera = mode in ["camera", "hybrid"]
        self.camera_interface = CameraSimulationInterface(use_camera=use_camera, table_size=table_size)
        
        # Simulation setup (if needed)
        self.physics_client = None
        self.ball_id = None
        self.table_id = None
        
        if mode in ["simulation", "hybrid"] and PYBULLET_AVAILABLE:
            self.setup_simulation()
        
        # State tracking
        self.table_pitch = 0.0
        self.table_roll = 0.0
        self.step_count = 0
        
        print(f"✅ Initialized ball balancing system in '{mode}' mode")
    
    def setup_simulation(self):
        """Setup PyBullet simulation (for simulation and hybrid modes)"""
        if not PYBULLET_AVAILABLE:
            return
            
        # Connect to PyBullet
        self.physics_client = p.connect(p.GUI)
        
        # Camera setup
        p.resetDebugVisualizerCamera(
            cameraDistance=0.6,
            cameraYaw=30,
            cameraPitch=-45,
            cameraTargetPosition=[0, 0, 0.06]
        )
        
        # Clean GUI
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        
        # Ground plane
        ground_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[2, 2, 0.01])
        ground_visual = p.createVisualShape(p.GEOM_BOX, halfExtents=[2, 2, 0.01], 
                                          rgbaColor=[0.4, 0.4, 0.4, 1])
        p.createMultiBody(baseMass=0, baseCollisionShapeIndex=ground_shape,
                         baseVisualShapeIndex=ground_visual, basePosition=[0, 0, -0.01])
        
        # Table
        table_half_size = self.table_size / 2
        table_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[table_half_size, table_half_size, 0.004])
        table_visual = p.createVisualShape(p.GEOM_BOX, halfExtents=[table_half_size, table_half_size, 0.004], 
                                         rgbaColor=[0.1, 0.1, 0.1, 1])
        self.table_start_pos = [0, 0, 0.06]
        self.table_id = p.createMultiBody(1.0, table_shape, table_visual, self.table_start_pos)
        
        # Ball (only in simulation mode)
        if self.mode == "simulation":
            ball_radius = 0.02
            self.ball_id = p.createMultiBody(
                baseMass=0.0027,
                baseCollisionShapeIndex=p.createCollisionShape(p.GEOM_SPHERE, radius=ball_radius),
                baseVisualShapeIndex=p.createVisualShape(p.GEOM_SPHERE, radius=ball_radius, 
                                                       rgbaColor=[0.9, 0.9, 0.9, 1]),
                basePosition=[0.06, 0.08, 0.5]  # Start off-center
            )
        
        print("✅ PyBullet simulation initialized")
    
    def calibrate_camera(self):
        """Calibrate camera for table detection"""
        if self.mode == "simulation":
            print("ℹ️ Camera calibration not needed in simulation mode")
            return True
            
        if hasattr(self.camera_interface.camera, 'calibrate_table_detection'):
            print("🎯 Starting camera calibration...")
            print("   Make sure the table is visible and well-lit")
            input("   Press Enter when ready...")
            return self.camera_interface.camera.calibrate_table_detection()
        else:
            print("❌ Camera not available for calibration")
            return False
    
    def run_control_loop(self, duration: float = 60.0):
        """
        Run the main control loop
        
        Args:
            duration: How long to run (seconds)
        """
        print(f"🎮 Starting control loop for {duration} seconds")
        print("   Press Ctrl+C to stop early")
        
        # Start camera tracking
        self.camera_interface.start_tracking()
        
        start_time = time.time()
        
        try:
            while time.time() - start_time < duration:
                # Get ball position and velocity
                ball_x, ball_y, ball_vx, ball_vy = self.camera_interface.get_ball_state(
                    simulation_ball_id=self.ball_id, 
                    pybullet_module=p if PYBULLET_AVAILABLE else None
                )
                
                # Calculate distance from center
                distance = np.sqrt(ball_x**2 + ball_y**2)
                
                # PID control
                pitch_angle = -self.pitch_pid.update(ball_y, 0.02)  # 50Hz control
                roll_angle = self.roll_pid.update(ball_x, 0.02)
                
                self.table_pitch = pitch_angle
                self.table_roll = roll_angle
                
                # Update table orientation (simulation mode)
                if self.table_id and PYBULLET_AVAILABLE:
                    quat = p.getQuaternionFromEuler([self.table_pitch, self.table_roll, 0])
                    p.resetBasePositionAndOrientation(self.table_id, self.table_start_pos, quat)
                
                # Step physics (simulation mode)
                if self.mode == "simulation" and PYBULLET_AVAILABLE:
                    for _ in range(12):  # 240Hz physics, 20Hz control
                        p.stepSimulation()
                
                # Print status
                if self.step_count % 25 == 0:  # Every ~0.5 seconds
                    mode_info = ""
                    if self.mode == "camera":
                        mode_info = " [CAMERA]"
                    elif self.mode == "hybrid":
                        mode_info = " [HYBRID]"
                    
                    print(f"Step {self.step_count:4d}: Pos=({ball_x:+.3f}, {ball_y:+.3f}) "
                          f"Vel=({ball_vx:+.3f}, {ball_vy:+.3f}) Dist={distance:.3f}m "
                          f"Angles=({self.table_pitch:+.3f}, {self.table_roll:+.3f}){mode_info}")
                
                # Check if ball fell off
                if distance > self.table_size / 2:
                    print(f"⚠️ Ball outside table boundary (distance: {distance:.3f}m)")
                    if self.mode == "simulation":
                        print("   Resetting ball...")
                        # Reset ball in simulation
                        if self.ball_id and PYBULLET_AVAILABLE:
                            p.resetBasePositionAndOrientation(self.ball_id, [0.06, 0.08, 0.5], [0, 0, 0, 1])
                
                self.step_count += 1
                time.sleep(0.05)  # 20Hz control loop
                
        except KeyboardInterrupt:
            print("\n⏹️ Control loop stopped by user")
        
        finally:
            self.camera_interface.stop_tracking()
            print("✅ Control loop finished")
    
    def test_camera_detection(self, duration: float = 10.0):
        """
        Test camera detection without control
        
        Args:
            duration: How long to test (seconds)
        """
        if self.mode == "simulation":
            print("ℹ️ Camera test not applicable in simulation mode")
            return
            
        print(f"📷 Testing camera detection for {duration} seconds")
        
        self.camera_interface.start_tracking()
        start_time = time.time()
        
        try:
            while time.time() - start_time < duration:
                ball_x, ball_y, ball_vx, ball_vy = self.camera_interface.get_ball_state()
                distance = np.sqrt(ball_x**2 + ball_y**2)
                
                print(f"Ball: ({ball_x:+.3f}, {ball_y:+.3f}) m, "
                      f"Velocity: ({ball_vx:+.3f}, {ball_vy:+.3f}) m/s, "
                      f"Distance: {distance:.3f}m")
                
                time.sleep(0.2)  # 5Hz reporting
                
        except KeyboardInterrupt:
            print("\n⏹️ Camera test stopped")
        
        finally:
            self.camera_interface.stop_tracking()
    
    def cleanup(self):
        """Clean up resources"""
        self.camera_interface.cleanup()
        
        if self.physics_client and PYBULLET_AVAILABLE:
            p.disconnect()
        
        print("✅ Cleanup completed")


def main():
    parser = argparse.ArgumentParser(description="Ball Balancing with Camera Integration")
    parser.add_argument("--mode", choices=["simulation", "camera", "hybrid"], 
                       default="simulation", help="Operating mode")
    parser.add_argument("--calibrate", action="store_true", 
                       help="Perform camera calibration before running")
    parser.add_argument("--test-camera", action="store_true",
                       help="Test camera detection only (no control)")
    parser.add_argument("--duration", type=float, default=60.0,
                       help="How long to run (seconds)")
    parser.add_argument("--table-size", type=float, default=0.25,
                       help="Table size in meters")
    
    args = parser.parse_args()
    
    print("🎯 Ball Balancing with Camera Integration")
    print("=" * 40)
    print(f"Mode: {args.mode}")
    print(f"Table size: {args.table_size}m")
    
    # Create system
    system = BallBalanceWithCamera(mode=args.mode, table_size=args.table_size)
    
    try:
        # Calibration step
        if args.calibrate and args.mode in ["camera", "hybrid"]:
            if not system.calibrate_camera():
                print("❌ Calibration failed, exiting")
                return
        
        # Test or run
        if args.test_camera:
            system.test_camera_detection(args.duration)
        else:
            system.run_control_loop(args.duration)
            
    except Exception as e:
        print(f"❌ Error: {e}")
    
    finally:
        system.cleanup()


if __name__ == "__main__":
    main()
