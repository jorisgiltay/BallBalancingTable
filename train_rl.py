import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold, CheckpointCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor
import os
import torch
import threading
import time
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


def train_rl_agent(use_early_stopping=True, use_curriculum=False, render_training=False, control_freq=50):
    """Train a reinforcement learning agent for ball balancing"""
    
    # Create environment with optional rendering and specified control frequency
    if render_training:
        env = BallBalanceEnv(render_mode="human", control_freq=control_freq)
        print("🎬 Training with visual rendering enabled")
    else:
        env = BallBalanceEnv(render_mode="rgb_array", control_freq=control_freq)  # No visual rendering for speed
        print("⚡ Training without visual rendering for maximum speed")
    print(f"⏱️ Control frequency: {control_freq} Hz")
    env = Monitor(env)
    
    # Create evaluation environment (without rendering for speed)
    eval_env = BallBalanceEnv(render_mode="rgb_array", control_freq=control_freq)
    eval_env = Monitor(eval_env)
    
    # Create model with ANTI-OVERFITTING hyperparameters
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=1e-4,     # Lower learning rate for stability
        n_steps=2048,           # Smaller buffer for more diverse experiences
        batch_size=64,          # Smaller batches to prevent overfitting
        n_epochs=3,             # Fewer epochs to prevent overfitting
        gamma=0.99,             # Standard discount factor
        gae_lambda=0.95,        # Keep GAE lambda
        clip_range=0.15,        # Moderate clipping
        clip_range_vf=0.15,     # Also clip value function
        ent_coef=0.02,          # Higher entropy to prevent overfitting
        vf_coef=0.5,            # Standard value function coefficient
        max_grad_norm=0.5,      # Standard gradient clipping
        tensorboard_log="./tensorboard_logs/",
        policy_kwargs=dict(
            net_arch=[64, 64],  # Smaller network to prevent overfitting
            activation_fn=torch.nn.Tanh,  # More stable than ReLU
        )
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
        reward_threshold = 1200.0 if control_freq >= 50 else 1000.0  # Much higher threshold for linear rewards
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
    
    # Create models directory
    os.makedirs("models", exist_ok=True)
    os.makedirs("tensorboard_logs", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)
    
    print("Starting training...")
    print("📊 Monitor training: tensorboard --logdir=./tensorboard_logs/")
    print("💾 Checkpoints saved every 10k steps to ./checkpoints/")
    print("🏆 Best models saved to ./models/")
    
    # Train the model - increased timesteps since agent was just starting to improve at 150k
    model.learn(
        total_timesteps=500000,  # Back to 500k - agent needs more time to fully converge
        callback=callbacks,
        progress_bar=True
    )
    
    # Save the final model
    model.save("models/ball_balance_ppo_final")
    print("Training completed! Model saved to models/ball_balance_ppo_final.zip")
    
    env.close()
    eval_env.close()
    
    return model


def test_trained_agent(model_path="models/ball_balance_ppo_final", control_freq=50):
    """Test a trained agent"""
    
    # Load the model
    model = PPO.load(model_path)
    
    # Create test environment with rendering and matching control frequency
    env = BallBalanceEnv(render_mode="human", control_freq=control_freq)
    print(f"Testing with {control_freq} Hz control frequency")
    
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
    parser.add_argument("--model", default="models/ball_balance_ppo_final", help="Path to model for testing")
    parser.add_argument("--no-early-stop", action="store_true", help="Disable early stopping during training")
    parser.add_argument("--resume-from", type=str, help="Resume training from specific checkpoint")
    parser.add_argument("--render", action="store_true", help="Enable visual rendering during training (slower)")
    parser.add_argument("--no-render", action="store_true", help="Disable visual rendering during training (faster)")
    parser.add_argument("--freq", type=int, default=50, help="Control frequency in Hz (default: 50)")
    
    args = parser.parse_args()
    
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
                          control_freq=args.freq)
    elif args.mode == "recover":
        from recovery_tool import rollback_to_checkpoint
        rollback_to_checkpoint()
    else:
        test_trained_agent(args.model, control_freq=args.freq)
