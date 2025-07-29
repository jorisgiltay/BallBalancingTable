import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.monitor import Monitor
import os
from ball_balance_env import BallBalanceEnv


def train_rl_agent():
    """Train a reinforcement learning agent for ball balancing"""
    
    # Create environment
    env = BallBalanceEnv(render_mode="human")
    env = Monitor(env)
    
    # Create evaluation environment (without rendering for speed)
    eval_env = BallBalanceEnv(render_mode="rgb_array")
    eval_env = Monitor(eval_env)
    
    # Create model with better hyperparameters for this problem
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=1024,  # Reduced from 2048
        batch_size=32,  # Reduced from 64
        n_epochs=4,     # Reduced from 10
        gamma=0.995,    # Increased from 0.99 for longer-term thinking
        gae_lambda=0.95,
        clip_range=0.1,  # Reduced from 0.2 for more stable updates
        ent_coef=0.01,   # Added entropy for exploration
        vf_coef=0.5,     # Value function coefficient
        max_grad_norm=0.5,  # Gradient clipping
        tensorboard_log="./tensorboard_logs/",
        policy_kwargs=dict(
            net_arch=dict(pi=[64, 64], vf=[64, 64])  # Smaller network
        )
    )
    
    # Create callbacks
    # Stop training when reward threshold is reached
    callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=0, verbose=1)
    eval_callback = EvalCallback(
        eval_env,
        callback_on_new_best=callback_on_best,
        eval_freq=5000,
        best_model_save_path="./models/",
        verbose=1
    )
    
    # Create models directory
    os.makedirs("models", exist_ok=True)
    os.makedirs("tensorboard_logs", exist_ok=True)
    
    print("Starting training...")
    print("You can monitor training with: tensorboard --logdir=./tensorboard_logs/")
    
    # Train the model
    model.learn(
        total_timesteps=100000,
        callback=eval_callback,
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
    parser.add_argument("--mode", choices=["train", "test"], default="train", help="Train or test the agent")
    parser.add_argument("--model", default="models/ball_balance_ppo_final", help="Path to model for testing")
    
    args = parser.parse_args()
    
    if args.mode == "train":
        train_rl_agent()
    else:
        test_trained_agent(args.model)
