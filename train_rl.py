import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
import os
import torch
from ball_balance_env import BallBalanceEnv


def train_rl_agent(use_early_stopping=True, use_curriculum=False):
    """Train a reinforcement learning agent for ball balancing"""
    
    # Create environment
    env = BallBalanceEnv(render_mode="human")
    env = Monitor(env)
    
    # Create evaluation environment (without rendering for speed)
    eval_env = BallBalanceEnv(render_mode="rgb_array")
    eval_env = Monitor(eval_env)
    
    # Create model with stability-focused hyperparameters
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=1e-4,  # Reduced for stability
        n_steps=2048,        # Increased for more experience per update
        batch_size=64,       # Increased for more stable gradients
        n_epochs=3,          # Reduced to prevent overfitting
        gamma=0.99,          # Standard discount factor
        gae_lambda=0.95,
        clip_range=0.1,      # Conservative clipping
        ent_coef=0.02,       # Higher entropy for continued exploration
        vf_coef=0.5,         
        max_grad_norm=0.5,   # Gradient clipping
        tensorboard_log="./tensorboard_logs/",
        policy_kwargs=dict(
            net_arch=dict(pi=[128, 128], vf=[128, 128]),  # Larger network for stability
            activation_fn=torch.nn.Tanh  # More stable activation
        )
    )
    
    # Create callbacks
    callbacks = []
    
    # Checkpoint callback - saves model every 10k steps
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path="./checkpoints/",
        name_prefix="ball_balance_checkpoint"
    )
    callbacks.append(checkpoint_callback)
    
    # Always add evaluation callback
    if use_early_stopping:
        # Create early stopping callback first
        callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=1200.0, verbose=1)
        
        # Create eval callback with early stopping
        eval_callback = EvalCallback(
            eval_env,
            eval_freq=5000,
            best_model_save_path="./models/",
            verbose=1,
            deterministic=True,
            render=False,
            callback_on_new_best=callback_on_best
        )
        print("Early stopping enabled - training will stop when reward reaches 1200.0")
    else:
        # Create eval callback without early stopping
        eval_callback = EvalCallback(
            eval_env,
            eval_freq=5000,
            best_model_save_path="./models/",
            verbose=1,
            deterministic=True,
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
    
    # Train the model
    model.learn(
        total_timesteps=100000,
        callback=callbacks,
        progress_bar=True
    )
    
    # Save the final model
    model.save("models/ball_balance_ppo_final")
    print("Training completed! Model saved to models/ball_balance_ppo_final.zip")
    
    env.close()
    eval_env.close()
    
    return model


def test_trained_agent(model_path="models/ball_balance_ppo_final"):
    """Test a trained agent"""
    
    # Load the model
    model = PPO.load(model_path)
    
    # Create test environment with rendering
    env = BallBalanceEnv(render_mode="human")
    
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
    
    args = parser.parse_args()
    
    if args.mode == "train":
        if args.resume_from:
            from recovery_tool import resume_training_from_checkpoint
            resume_training_from_checkpoint(args.resume_from)
        else:
            train_rl_agent(use_early_stopping=not args.no_early_stop)
    elif args.mode == "recover":
        from recovery_tool import rollback_to_checkpoint
        rollback_to_checkpoint()
    else:
        test_trained_agent(args.model)
