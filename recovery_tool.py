import os
import glob
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from ball_balance_env import BallBalanceEnv


def find_best_checkpoint():
    """Find the best performing checkpoint based on filename"""
    checkpoint_files = glob.glob("checkpoints/ball_balance_checkpoint_*_steps.zip")
    
    if not checkpoint_files:
        print("No checkpoints found in ./checkpoints/")
        return None
    
    # Sort by step number (higher = more recent)
    checkpoint_files.sort(key=lambda x: int(x.split('_')[-2]))
    
    print("Available checkpoints:")
    for i, ckpt in enumerate(checkpoint_files[-5:]):  # Show last 5
        steps = ckpt.split('_')[-2]
        print(f"  {i+1}. {ckpt} ({steps} steps)")
    
    return checkpoint_files


def resume_training_from_checkpoint(checkpoint_path, additional_steps=50000):
    """Resume training from a specific checkpoint"""
    
    print(f"Loading model from: {checkpoint_path}")
    
    # Load the model
    model = PPO.load(checkpoint_path)
    
    # Create fresh environments
    env = BallBalanceEnv(render_mode="human")
    env = Monitor(env)
    
    eval_env = BallBalanceEnv(render_mode="rgb_array")
    eval_env = Monitor(eval_env)
    
    # Set the environment (required after loading)
    model.set_env(env)
    
    print(f"Resuming training for {additional_steps} more steps...")
    
    # Continue training
    model.learn(
        total_timesteps=additional_steps,
        reset_num_timesteps=False,  # Don't reset step counter
        progress_bar=True
    )
    
    # Save the continued model
    model.save("models/ball_balance_ppo_continued")
    print("Continued training saved to models/ball_balance_ppo_continued.zip")
    
    env.close()
    eval_env.close()
    
    return model


def rollback_to_checkpoint():
    """Interactive tool to rollback to a previous checkpoint when training goes bad"""
    
    checkpoints = find_best_checkpoint()
    if not checkpoints:
        return None
    
    print("\n🆘 TRAINING RECOVERY TOOL")
    print("=" * 40)
    print("Use this when your RL agent starts performing worse!")
    print()
    
    # Show recent checkpoints
    recent_checkpoints = checkpoints[-10:]  # Last 10 checkpoints
    
    print("Recent checkpoints:")
    for i, ckpt in enumerate(recent_checkpoints):
        steps = ckpt.split('_')[-2]
        print(f"  {i+1:2d}. {steps:>6s} steps - {os.path.basename(ckpt)}")
    
    print(f"  {len(recent_checkpoints)+1:2d}. Test all recent checkpoints")
    print(f"  {len(recent_checkpoints)+2:2d}. Cancel")
    
    try:
        choice = int(input("\nSelect checkpoint to load (number): ")) - 1
        
        if choice == len(recent_checkpoints):
            # Test all checkpoints
            test_all_checkpoints(recent_checkpoints[-5:])
            return None
        elif choice == len(recent_checkpoints) + 1:
            print("Cancelled.")
            return None
        elif 0 <= choice < len(recent_checkpoints):
            selected_checkpoint = recent_checkpoints[choice]
            
            print(f"\nDo you want to:")
            print("1. Just test this checkpoint")
            print("2. Resume training from this checkpoint")
            
            action = input("Choose (1 or 2): ")
            
            if action == "1":
                test_checkpoint_performance(selected_checkpoint)
            elif action == "2":
                resume_training_from_checkpoint(selected_checkpoint)
            
            return selected_checkpoint
        else:
            print("Invalid selection.")
            return None
            
    except (ValueError, KeyboardInterrupt):
        print("Cancelled.")
        return None


def test_checkpoint_performance(checkpoint_path, episodes=3):
    """Test how well a specific checkpoint performs"""
    
    print(f"\n🧪 Testing checkpoint: {os.path.basename(checkpoint_path)}")
    
    # Load model
    model = PPO.load(checkpoint_path)
    
    # Create test environment
    env = BallBalanceEnv(render_mode="human")
    
    total_rewards = []
    episode_lengths = []
    
    for episode in range(episodes):
        obs, info = env.reset()
        episode_reward = 0
        step_count = 0
        
        print(f"\nEpisode {episode + 1}/{episodes}")
        
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            step_count += 1
            
            if terminated or truncated:
                break
        
        total_rewards.append(episode_reward)
        episode_lengths.append(step_count)
        
        print(f"  Reward: {episode_reward:7.2f}, Steps: {step_count:4d}, Distance: {info['distance_from_center']:.3f}")
    
    print(f"\n📊 Performance Summary:")
    print(f"  Average Reward: {np.mean(total_rewards):7.2f} ± {np.std(total_rewards):.2f}")
    print(f"  Average Length: {np.mean(episode_lengths):7.1f} ± {np.std(episode_lengths):.1f}")
    
    env.close()
    
    return np.mean(total_rewards), np.mean(episode_lengths)


def test_all_checkpoints(checkpoint_list):
    """Test multiple checkpoints to find the best one"""
    
    print("\n🏆 CHECKPOINT COMPARISON")
    print("=" * 50)
    
    results = []
    
    for ckpt in checkpoint_list:
        print(f"\nTesting {os.path.basename(ckpt)}...")
        avg_reward, avg_length = test_checkpoint_performance(ckpt, episodes=2)
        results.append((ckpt, avg_reward, avg_length))
    
    # Sort by average reward
    results.sort(key=lambda x: x[1], reverse=True)
    
    print(f"\n🥇 CHECKPOINT RANKINGS:")
    print("-" * 50)
    for i, (ckpt, reward, length) in enumerate(results):
        steps = ckpt.split('_')[-2]
        print(f"{i+1}. {steps:>6s} steps: Reward {reward:7.2f}, Length {length:7.1f}")
    
    best_checkpoint = results[0][0]
    print(f"\n🏆 Best checkpoint: {os.path.basename(best_checkpoint)}")
    
    use_best = input("Resume training from best checkpoint? (y/n): ")
    if use_best.lower().startswith('y'):
        resume_training_from_checkpoint(best_checkpoint)


if __name__ == "__main__":
    import numpy as np
    
    print("🆘 RL Training Recovery Tool")
    print("=" * 40)
    print("Use this when your RL training starts failing!")
    print()
    
    rollback_to_checkpoint()
