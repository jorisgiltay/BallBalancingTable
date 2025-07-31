"""
Modified version of compare_control.py with camera integration support

This version allows switching between:
1. Pure simulation (original behavior)
2. Camera input with simulated physics (hybrid mode)
3. Camera input only (for real hardware)

Usage:
- python compare_control_camera.py --control pid --camera simulation
- python compare_control_camera.py --control rl --camera hybrid --visuals
- python compare_control_camera.py --control pid --camera real --calibrate
"""

import argparse
import pybullet as p
from compare_control import BallBalanceComparison
from camera_interface import CameraSimulationInterface


class BallBalanceComparisonWithCamera(BallBalanceComparison):
    """
    Extended version of BallBalanceComparison with camera support
    """
    
    def __init__(self, control_method="pid", control_freq=50, enable_visuals=False, camera_mode="simulation"):
        """
        Initialize with camera support
        
        Args:
            control_method: "pid" or "rl"
            control_freq: Control frequency in Hz
            enable_visuals: Enable matplotlib dashboard
            camera_mode: "simulation", "hybrid", or "real"
        """
        self.camera_mode = camera_mode
        
        # Initialize parent class
        super().__init__(control_method, control_freq, enable_visuals)
        
        # Setup camera interface
        use_camera = camera_mode in ["hybrid", "real"]
        self.camera_interface = CameraSimulationInterface(use_camera=use_camera, table_size=0.25)
        
        print(f"✅ Camera mode: {camera_mode}")
        
        if use_camera:
            print("📷 Camera interface initialized")
            if camera_mode == "real":
                print("⚠️ Real camera mode - make sure PyBullet simulation represents actual table")
    
    def calibrate_camera(self):
        """Calibrate camera for table detection"""
        if self.camera_mode == "simulation":
            print("ℹ️ Camera calibration not needed in simulation mode")
            return True
            
        if hasattr(self.camera_interface.camera, 'calibrate_table_detection'):
            print("🎯 Starting camera calibration...")
            print("   Position the camera to see the entire table")
            print("   Make sure the table is well-lit and no ball is present")
            input("   Press Enter when ready...")
            return self.camera_interface.camera.calibrate_table_detection()
        else:
            print("❌ Camera not available for calibration")
            return False
    
    def get_observation(self):
        """
        Get current state observation - now with camera support
        Compatible with existing RL interface
        """
        if self.camera_mode == "simulation":
            # Use original simulation-based observation
            return super().get_observation()
        
        # Get ball state from camera interface
        ball_x, ball_y, ball_vx, ball_vy = self.camera_interface.get_ball_state(
            simulation_ball_id=self.ball_id if hasattr(self, 'ball_id') else None,
            pybullet_module=p if self.camera_mode == "hybrid" else None
        )
        
        # Update previous position for next velocity calculation (maintain compatibility)
        self.prev_ball_pos = [ball_x, ball_y]
        
        # Return observation in same format as original
        import numpy as np
        observation = np.array([
            ball_x, ball_y, ball_vx, ball_vy, self.table_pitch, self.table_roll
        ], dtype=np.float32)
        
        return observation
    
    def setup_simulation(self):
        """Setup simulation - modified for camera modes"""
        if self.camera_mode == "real":
            # In real camera mode, don't create PyBullet simulation
            print("ℹ️ Real camera mode - skipping PyBullet simulation setup")
            return
        
        # Setup simulation normally for simulation and hybrid modes
        super().setup_simulation()
        
        if self.camera_mode == "hybrid":
            print("🔗 Hybrid mode - camera input with simulated physics")
    
    def reset_ball(self, position=None, randomize=True):
        """Reset ball - modified for camera modes"""
        if self.camera_mode == "real":
            print("ℹ️ Real camera mode - please manually position the ball on the table")
            input("   Press Enter when ball is positioned...")
            return
        
        # Normal reset for simulation and hybrid modes
        super().reset_ball(position, randomize)
    
    def run_simulation(self):
        """
        Main simulation loop - modified for camera support
        """
        # Start camera tracking if using camera
        if self.camera_mode in ["hybrid", "real"]:
            self.camera_interface.start_tracking()
            print("📷 Camera tracking started")
        
        try:
            # Run the original simulation loop
            super().run_simulation()
        finally:
            # Stop camera tracking
            if self.camera_mode in ["hybrid", "real"]:
                self.camera_interface.stop_tracking()
                print("📷 Camera tracking stopped")
            
            # Cleanup camera resources
            self.camera_interface.cleanup()


def main():
    parser = argparse.ArgumentParser(description="Ball Balancing Control Comparison with Camera Support")
    parser.add_argument("--control", choices=["pid", "rl"], default="pid", 
                       help="Control method to start with")
    parser.add_argument("--freq", type=int, default=50,
                       help="Control frequency in Hz (default: 50)")
    parser.add_argument("--visuals", action="store_true",
                       help="Enable visual dashboard")
    parser.add_argument("--camera", choices=["simulation", "hybrid", "real"], default="simulation",
                       help="Camera mode: simulation (no camera), hybrid (camera + physics), real (camera only)")
    parser.add_argument("--calibrate", action="store_true",
                       help="Perform camera calibration before starting")
    
    args = parser.parse_args()
    
    print("🎯 Ball Balancing Control Comparison with Camera Support")
    print("=" * 55)
    print(f"Control method: {args.control}")
    print(f"Control frequency: {args.freq} Hz")
    print(f"Camera mode: {args.camera}")
    print(f"Visual dashboard: {'Enabled' if args.visuals else 'Disabled'}")
    
    # Create simulator with camera support
    simulator = BallBalanceComparisonWithCamera(
        control_method=args.control, 
        control_freq=args.freq, 
        enable_visuals=args.visuals,
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
            print("   4. Control outputs will need to be connected to actual servos")
            input("\n   Press Enter to continue...")
        elif args.camera == "hybrid":
            print("\n📋 Hybrid Mode Instructions:")
            print("   1. RealSense camera provides ball position")
            print("   2. PyBullet simulates physics and visualizes control")
            print("   3. Great for testing camera integration before hardware deployment")
            input("\n   Press Enter to continue...")
        
        # Run simulation
        simulator.run_simulation()
        
    except KeyboardInterrupt:
        print("\n⏹️ Simulation stopped by user")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup is handled in run_simulation()
        pass


if __name__ == "__main__":
    main()
