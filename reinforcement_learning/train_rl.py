import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold, CheckpointCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor
import os
import torch
import threading
import time
import subprocess
from ball_balance_env import BallBalanceEnv


class RenderToggleCallback(BaseCallback):
    """Callback that allows toggling rendering during training"""
    
    def __init__(self, env, verbose=0):
        super().__init__(verbose)
        self.env = env
        self.render_enabled = False
        self.listener_thread = None
        self.running = True
        self.should_stop = False
        
    def _on_training_start(self) -> None:
        print("\n🎮 Rendering Controls:")
        print("  Press 'r' + Enter to toggle rendering ON/OFF")
        print("  Press 'q' + Enter to quit training")
        print("  Current rendering: OFF (for speed)")
        
        # Start keyboard listener in separate thread
        self.listener_thread = threading.Thread(target=self._keyboard_listener, daemon=True)
        self.listener_thread.start()
        
    def _keyboard_listener(self):
        """Listen for keyboard input in separate thread"""
        while self.running:
            try:
                key = input().strip().lower()
                if key == 'r':
                    self._toggle_rendering()
                elif key == 'q':
                    print("🛑 Stopping training...")
                    # Set a flag that will be checked in _on_step
                    self.should_stop = True
                    break
            except (EOFError, KeyboardInterrupt):
                break
                
    def _toggle_rendering(self):
        """Toggle rendering mode"""
        try:
            self.render_enabled = not self.render_enabled
            
            # Access the underlying BallBalanceEnv through the Monitor wrapper
            underlying_env = self.env
            if hasattr(self.env, 'env'):  # If it's wrapped in Monitor
                underlying_env = self.env.env
            
            if self.render_enabled:
                # Switch to human rendering
                underlying_env.render_mode = "human"
                print("🎬 Rendering ENABLED - You can now see the training!")
            else:
                # Switch to rgb_array (no visual)
                underlying_env.render_mode = "rgb_array"
                print("⚡ Rendering DISABLED - Training at full speed")
                
        except Exception as e:
            print(f"Error toggling rendering: {e}")
            print("Rendering toggle may not be supported with this environment wrapper")
    
    def _on_training_end(self) -> None:
        self.running = False
    
    def _on_step(self) -> bool:
        """Called after each step. Must return True to continue training."""
        # Return False to stop training if user pressed 'q'
        if self.should_stop:
            print("🏁 Training stopped by user request")
            return False
        return True


def train_rl_agent(use_early_stopping=True, use_curriculum=False, render_training=False, control_freq=50, start_tensorboard=False, enable_servo_uncertainty=True):
    """Train a reinforcement learning agent for ball balancing"""
    
    # Create directories first (in local reinforcement_learning directory)
    os.makedirs("models", exist_ok=True)
    os.makedirs("tensorboard_logs", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)
    
    # Start TensorBoard automatically if requested
    tensorboard_process = None
    if start_tensorboard:
        import subprocess
        import webbrowser
        try:
            print("🚀 Starting TensorBoard...")
            tensorboard_process = subprocess.Popen([
                "tensorboard", 
                "--logdir=./tensorboard_logs/", 
                "--port=6006",
                "--host=localhost"
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
            # Give TensorBoard a moment to start
            time.sleep(2)
            
            # Try to open browser automatically
            try:
                webbrowser.open("http://localhost:6006")
                print("📊 TensorBoard started at http://localhost:6006 (opened in browser)")
            except:
                print("📊 TensorBoard started at http://localhost:6006")
                
        except FileNotFoundError:
            print("⚠️ TensorBoard not found. Install with: pip install tensorboard")
            print("📊 You can still monitor logs manually: tensorboard --logdir=./tensorboard_logs/")
        except Exception as e:
            print(f"⚠️ Failed to start TensorBoard automatically: {e}")
            print("📊 You can start it manually: tensorboard --logdir=./tensorboard_logs/")
    
    # Create environment with optional rendering and specified control frequency
    if render_training:
        env = BallBalanceEnv(render_mode="human", control_freq=control_freq, enable_servo_uncertainty=enable_servo_uncertainty)
        print("🎬 Training with visual rendering enabled")
    else:
        env = BallBalanceEnv(render_mode="rgb_array", control_freq=control_freq, enable_servo_uncertainty=enable_servo_uncertainty)  # No visual rendering for speed
        print("⚡ Training without visual rendering for maximum speed")
    print(f"⏱️ Control frequency: {control_freq} Hz")
    print(f"🔧 Servo uncertainty: {'ENABLED (realistic)' if enable_servo_uncertainty else 'DISABLED (perfect control)'}")
    env = Monitor(env)
    
    # Create evaluation environment (without rendering for speed)
    eval_env = BallBalanceEnv(render_mode="rgb_array", control_freq=control_freq, enable_servo_uncertainty=enable_servo_uncertainty)
    eval_env = Monitor(eval_env)
    
    # Create model with ANTI-OVERFITTING hyperparameters
    model = SAC(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=1e-4,          # Lower LR for stability
        buffer_size=100000,          # Replay buffer size
        batch_size=64,               # Batch size for updates
        tau=0.005,                   # Target network smoothing coefficient
        gamma=0.99,                  # Discount factor
        train_freq=1,                # Train every environment step
        gradient_steps=1,            # Number of updates per step
        ent_coef="auto",             # Learn entropy coefficient automatically
        target_update_interval=1,    # How often to update target net
        learning_starts=1000,        # Delay training until some experience
        policy_kwargs=dict(
            net_arch=[64, 64],       # Smaller network
            activation_fn=torch.nn.Tanh  # More stable than ReLU
        ),
        tensorboard_log="./tensorboard_logs/"
    )
    
    
    # Create callbacks
    callbacks = []
    
    # Add render toggle callback for interactive control
    render_toggle_callback = RenderToggleCallback(env, verbose=1)
    callbacks.append(render_toggle_callback)
    
    # Checkpoint callback - saves model every 10k steps
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path="./checkpoints/",
        name_prefix="ball_balance_checkpoint"
    )
    callbacks.append(checkpoint_callback)
    
    # Always add evaluation callback
    if use_early_stopping:
        # Adjusted thresholds for simplified linear reward function
        # With max ~1800 per episode, aim for 70-80% of theoretical max
        reward_threshold = 1800 if control_freq >= 50 else 1000.0  # Much higher threshold for linear rewards
        callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=reward_threshold, verbose=1)
        print(f"Early stopping threshold: {reward_threshold} (adjusted for simplified linear reward)")
        
        # Create eval callback with early stopping - FIXED: use stochastic evaluation
        eval_callback = EvalCallback(
            eval_env,
            eval_freq=2500,  # More frequent evaluation
            best_model_save_path="./models/",
            verbose=1,
            deterministic=False,  # CHANGED: Allow some randomness in evaluation
            render=False,
            callback_on_new_best=callback_on_best
        )
        print(f"Early stopping enabled - training will stop when reward reaches {reward_threshold}")
    else:
        # Create eval callback without early stopping - FIXED: use stochastic evaluation
        eval_callback = EvalCallback(
            eval_env,
            eval_freq=2500,  # More frequent evaluation
            best_model_save_path="./models/",
            verbose=1,
            deterministic=False,  # CHANGED: Allow some randomness in evaluation
            render=False
        )
        print("Early stopping disabled - training will run for full duration")
    
    callbacks.append(eval_callback)
    
    print("Starting training...")
    if not start_tensorboard:
        print("📊 Monitor training: tensorboard --logdir=./tensorboard_logs/")
    print("💾 Checkpoints saved every 10k steps to ./checkpoints/")
    print("🏆 Best models saved to ./models/")
    
    # Train the model - increased timesteps since agent was just starting to improve at 150k
    try:
        model.learn(
            total_timesteps=750000,  # Back to 500k - agent needs more time to fully converge
            callback=callbacks,
            progress_bar=True
        )
    finally:
        # Clean up TensorBoard process if it was started
        if tensorboard_process:
            try:
                tensorboard_process.terminate()
                print("🛑 TensorBoard stopped")
            except:
                pass
    
    # Save the final model
    model.save("./models/ball_balance_ppo_final")
    print("Training completed! Model saved to ./models/ball_balance_ppo_final.zip")
    
    env.close()
    eval_env.close()
    
    return model


def test_trained_agent(model_path="./models/ball_balance_ppo_final", control_freq=50, enable_servo_uncertainty=True):
    """Test a trained agent"""
    
    # Load the model
    model = PPO.load(model_path)
    
    # Create test environment with rendering and matching control frequency
    env = BallBalanceEnv(render_mode="human", control_freq=control_freq, enable_servo_uncertainty=enable_servo_uncertainty)
    print(f"Testing with {control_freq} Hz control frequency")
    print(f"🔧 Servo uncertainty: {'ENABLED (realistic)' if enable_servo_uncertainty else 'DISABLED (perfect control)'}")
    
    print("Testing trained agent. Press Ctrl+C to stop.")
    
    try:
        for episode in range(10):
            obs, info = env.reset()
            episode_reward = 0
            step_count = 0
            
            while True:
                action, _states = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                episode_reward += reward
                step_count += 1
                
                if terminated or truncated:
                    print(f"Episode {episode + 1}: Steps: {step_count}, Reward: {episode_reward:.2f}, Distance: {info['distance_from_center']:.3f}")
                    break
    
    except KeyboardInterrupt:
        print("Testing stopped by user")
    
    env.close()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train or test RL agent for ball balancing")
    parser.add_argument("--mode", choices=["train", "test", "recover"], default="train", help="Train, test, or recover from checkpoint")
    parser.add_argument("--model", default="./models/ball_balance_ppo_final", help="Path to model for testing")
    parser.add_argument("--no-early-stop", action="store_true", help="Disable early stopping during training")
    parser.add_argument("--resume-from", type=str, help="Resume training from specific checkpoint")
    parser.add_argument("--render", action="store_true", help="Enable visual rendering during training (slower)")
    parser.add_argument("--no-render", action="store_true", help="Disable visual rendering during training (faster)")
    parser.add_argument("--freq", type=int, default=60, help="Control frequency in Hz (default: 60)")
    parser.add_argument("--tensorboard", action="store_true", help="Automatically start TensorBoard during training")
    parser.add_argument("--no-servo-uncertainty", action="store_true", help="Disable servo uncertainty for perfect control (debugging)")
    parser.add_argument("--servo-uncertainty", action="store_true", help="Enable servo uncertainty for realistic training (default)")
    
    args = parser.parse_args()
    
    # Determine servo uncertainty setting
    if args.no_servo_uncertainty and args.servo_uncertainty:
        print("Error: Cannot use both --servo-uncertainty and --no-servo-uncertainty")
        exit(1)
    elif args.no_servo_uncertainty:
        servo_uncertainty = False
        print("🔧 Servo uncertainty: DISABLED (perfect control for debugging)")
    else:
        servo_uncertainty = True  # Default to enabled for realistic training
        print("🔧 Servo uncertainty: ENABLED (realistic XL430-250T behavior)")
    
    # Determine render mode
    if args.render and args.no_render:
        print("Error: Cannot use both --render and --no-render")
        exit(1)
    elif args.render:
        render_training = True
    elif args.no_render:
        render_training = False
    else:
        # Default: no rendering for speed
        render_training = False
    
    if args.mode == "train":
        if args.resume_from:
            from recovery_tool import resume_training_from_checkpoint
            resume_training_from_checkpoint(args.resume_from)
        else:
            train_rl_agent(use_early_stopping=not args.no_early_stop, 
                          render_training=render_training, 
                          control_freq=args.freq,
                          start_tensorboard=args.tensorboard,
                          enable_servo_uncertainty=servo_uncertainty)
    elif args.mode == "recover":
        from recovery_tool import rollback_to_checkpoint
        rollback_to_checkpoint()
    else:
        test_trained_agent(args.model, control_freq=args.freq, enable_servo_uncertainty=servo_uncertainty)
