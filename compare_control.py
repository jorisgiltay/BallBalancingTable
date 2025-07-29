import pybullet as p
import pybullet_data
import time
import numpy as np
from pid_controller import PIDController
import argparse
import os


class BallBalanceComparison:
    """
    Compare PID control vs Reinforcement Learning control
    """
    
    def __init__(self, control_method="pid"):
        self.control_method = control_method
        
        # PID controllers (create these BEFORE setup_simulation)
        self.pitch_pid = PIDController(kp=10.0, ki=0.1, kd=2.0, output_limits=(-0.05, 0.05))
        self.roll_pid = PIDController(kp=10.0, ki=0.1, kd=2.0, output_limits=(-0.05, 0.05))
        
        self.setup_simulation()
        
        # RL model (will be loaded if using RL)
        self.rl_model = None
        if control_method == "rl":
            self.load_rl_model()
        
        # State tracking
        self.table_pitch = 0.0
        self.table_roll = 0.0
        self.step_count = 0
        self.randomize_ball = True  # Start with randomized positions
        
    def setup_simulation(self):
        """Initialize PyBullet simulation"""
        p.connect(p.GUI)
        
        # Set up a better camera view
        p.resetDebugVisualizerCamera(
            cameraDistance=0.8,        # Distance from target
            cameraYaw=45,              # Horizontal angle (degrees)
            cameraPitch=-30,           # Vertical angle (degrees, negative = looking down)
            cameraTargetPosition=[0, 0, 0.06]  # Look at the table center
        )
        # Optional: configure visualizer for cleaner view
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1)  # Keep GUI
        p.configureDebugVisualizer(p.COV_ENABLE_TINY_RENDERER, 0)  # Better rendering
        
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
        table_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.25, 0.25, 0.004])
        table_visual = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.25, 0.25, 0.004], rgbaColor=[0.0, 0.0, 0.0, 1])
        self.table_start_pos = [0, 0, 0.06]
        self.table_id = p.createMultiBody(1.0, table_shape, table_visual, self.table_start_pos)
        
        # Ball
        self.ball_radius = 0.02
        self.reset_ball()
        
        self.dt = 1.0 / 240.0
        
    def reset_ball(self, position=None, randomize=True):
        """Reset ball to initial position"""
        if position is None:
            if randomize:
                # Random position like in RL training
                position = [
                    np.random.uniform(-0.15, 0.15),
                    np.random.uniform(-0.15, 0.15),
                    0.5
                ]
                print(f"Ball reset to random position: ({position[0]:.2f}, {position[1]:.2f})")
            else:
                # Fixed position for consistent testing
                position = [0.12, 0.15, 0.5]  # Default position
                print(f"Ball reset to fixed position: ({position[0]:.2f}, {position[1]:.2f})")
            
        if hasattr(self, 'ball_id'):
            p.removeBody(self.ball_id)
            
        self.ball_id = p.createMultiBody(
            baseMass=0.1,
            baseCollisionShapeIndex=p.createCollisionShape(p.GEOM_SPHERE, radius=self.ball_radius),
            baseVisualShapeIndex=p.createVisualShape(p.GEOM_SPHERE, radius=self.ball_radius, rgbaColor=[1, 0, 0, 1]),
            basePosition=position
        )
        
        # Reset PID controllers
        self.pitch_pid.reset()
        self.roll_pid.reset()
        self.table_pitch = 0.0
        self.table_roll = 0.0
        self.step_count = 0
        
        # Let ball settle
        for _ in range(100):
            p.stepSimulation()
    
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
        """Get current state observation for RL - POSITION ONLY like PID"""
        # Ball position and orientation
        ball_pos, _ = p.getBasePositionAndOrientation(self.ball_id)
        ball_x, ball_y, ball_z = ball_pos
        
        # POSITION ONLY - no velocity for fair comparison with PID
        observation = np.array([
            ball_x, ball_y, self.table_pitch, self.table_roll
        ], dtype=np.float32)
        
        return observation
    
    def pid_control(self, ball_x, ball_y):
        """PID control logic"""
        pitch_angle = -self.pitch_pid.update(ball_y, self.dt)
        roll_angle = self.roll_pid.update(ball_x, self.dt)
        return pitch_angle, roll_angle
    
    def rl_control(self, observation):
        """RL control logic"""
        if self.rl_model is None:
            return 0.0, 0.0
        
        action, _ = self.rl_model.predict(observation, deterministic=True)
        print(f"RL action: {action}")
        return action[0], action[1]  # pitch_change, roll_change
    
    def run_simulation(self):
        """Main simulation loop"""
        print(f"Running simulation with {self.control_method.upper()} control")
        print("Controls:")
        print("  'r' - Reset ball")
        print("  'f' - Toggle fixed/random ball position")
        print("  'p' - Switch to PID control")
        print("  'l' - Switch to RL control") 
        print("  'q' - Quit")
        print(f"Ball position mode: {'Random' if self.randomize_ball else 'Fixed'}")
        
        while True:
            # Get ball position
            ball_pos, _ = p.getBasePositionAndOrientation(self.ball_id)
            ball_x, ball_y, ball_z = ball_pos
            
            # Control logic
            if self.control_method == "pid":
                pitch_angle, roll_angle = self.pid_control(ball_x, ball_y)
                # For PID, these are absolute angles
                self.table_pitch = pitch_angle
                self.table_roll = roll_angle
            else:  # RL
                observation = self.get_observation()
                pitch_change, roll_change = self.rl_control(observation)
                # For RL, these are angle changes
                self.table_pitch += pitch_change
                self.table_roll += roll_change
                # Clip to reasonable limits
                self.table_pitch = np.clip(self.table_pitch, -0.1, 0.1)
                self.table_roll = np.clip(self.table_roll, -0.1, 0.1)
            
            # Update table orientation
            quat = p.getQuaternionFromEuler([self.table_pitch, self.table_roll, 0])
            p.resetBasePositionAndOrientation(self.table_id, self.table_start_pos, quat)
            
            # Handle keyboard input
            keys = p.getKeyboardEvents()
            for key, state in keys.items():
                if state & p.KEY_WAS_TRIGGERED:
                    if key == ord('r'):
                        print("Resetting ball...")
                        self.reset_ball(randomize=self.randomize_ball)
                    elif key == ord('f'):
                        self.randomize_ball = not self.randomize_ball
                        mode = "Random" if self.randomize_ball else "Fixed"
                        print(f"Ball position mode: {mode}")
                        self.reset_ball(randomize=self.randomize_ball)
                    elif key == ord('p'):
                        print("Switching to PID control")
                        self.control_method = "pid"
                        self.reset_ball(randomize=self.randomize_ball)
                    elif key == ord('l'):
                        print("Switching to RL control")
                        if self.rl_model is not None:
                            self.control_method = "rl"
                            self.reset_ball(randomize=self.randomize_ball)
                        else:
                            print("RL model not available. Attempting to load...")
                            self.load_rl_model()
                            if self.rl_model is not None:
                                self.control_method = "rl"
                                self.reset_ball(randomize=self.randomize_ball)
                                print("✅ RL control activated!")
                            else:
                                print("❌ Still no RL model available")
                    elif key == ord('q'):
                        print("Quitting...")
                        return
            
            # Check if ball fell off
            distance_from_center = np.sqrt(ball_x**2 + ball_y**2)
            if distance_from_center > 0.25 or ball_z < 0.05:
                print(f"Ball fell off after {self.step_count} steps. Resetting...")
                self.reset_ball(randomize=self.randomize_ball)
            
            # Print status every 240 steps (1 second)
            if self.step_count % 240 == 0:
                print(f"Step: {self.step_count}, Method: {self.control_method.upper()}, "
                      f"Ball pos: ({ball_x:.3f}, {ball_y:.3f}), "
                      f"Distance: {distance_from_center:.3f}, "
                      f"Table: ({self.table_pitch:.3f}, {self.table_roll:.3f})")
            
            p.stepSimulation()
            time.sleep(self.dt)
            self.step_count += 1


def main():
    parser = argparse.ArgumentParser(description="Ball Balancing Control Comparison")
    parser.add_argument("--control", choices=["pid", "rl"], default="pid", 
                       help="Control method to start with")
    
    args = parser.parse_args()
    
    simulator = BallBalanceComparison(control_method=args.control)
    
    try:
        simulator.run_simulation()
    except KeyboardInterrupt:
        print("\nSimulation stopped by user")
    finally:
        p.disconnect()


if __name__ == "__main__":
    main()
