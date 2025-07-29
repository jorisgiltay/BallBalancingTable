import numpy as np
import matplotlib.pyplot as plt
from ball_balance_env import BallBalanceEnv
import time


def test_reward_function():
    """Test the reward function with different ball positions"""
    env = BallBalanceEnv(render_mode="rgb_array")
    
    # Test different positions
    positions = [
        [0.0, 0.0],    # Center
        [0.05, 0.0],   # Slightly off center
        [0.1, 0.0],    # Further off
        [0.15, 0.0],   # Near edge
        [0.2, 0.0],    # Very near edge
        [0.25, 0.0],   # At edge (should fail)
    ]
    
    print("Testing Reward Function:")
    print("Position (x,y) -> Reward")
    print("-" * 30)
    
    for pos in positions:
        # Create a fake observation
        observation = np.array([pos[0], pos[1], 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        reward = env._calculate_reward(observation)
        distance = np.sqrt(pos[0]**2 + pos[1]**2)
        print(f"({pos[0]:4.2f}, {pos[1]:4.2f}) -> {reward:6.3f} (dist: {distance:.3f})")
    
    env.close()


def test_random_policy(episodes=5):
    """Test random actions to see baseline performance"""
    env = BallBalanceEnv(render_mode="human")
    
    print(f"\nTesting Random Policy for {episodes} episodes:")
    print("-" * 50)
    
    total_rewards = []
    episode_lengths = []
    
    for episode in range(episodes):
        obs, info = env.reset()
        episode_reward = 0
        step_count = 0
        
        while True:
            # Random action
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            step_count += 1
            
            if terminated or truncated:
                break
        
        total_rewards.append(episode_reward)
        episode_lengths.append(step_count)
        print(f"Episode {episode+1}: Steps: {step_count:3d}, Reward: {episode_reward:7.2f}, Distance: {info['distance_from_center']:.3f}")
    
    print(f"\nRandom Policy Summary:")
    print(f"Average Reward: {np.mean(total_rewards):.2f} ± {np.std(total_rewards):.2f}")
    print(f"Average Length: {np.mean(episode_lengths):.1f} ± {np.std(episode_lengths):.1f}")
    
    env.close()
    return total_rewards, episode_lengths


def visualize_starting_positions():
    """Show the randomization of starting positions"""
    import matplotlib.pyplot as plt
    
    print("Generating 100 random starting positions...")
    
    positions_x = []
    positions_y = []
    
    for _ in range(100):
        x = np.random.uniform(-0.15, 0.15)
        y = np.random.uniform(-0.15, 0.15)
        positions_x.append(x)
        positions_y.append(y)
    
    plt.figure(figsize=(8, 8))
    plt.scatter(positions_x, positions_y, alpha=0.6, s=50)
    
    # Draw table boundary
    table_x = [-0.25, 0.25, 0.25, -0.25, -0.25]
    table_y = [-0.25, -0.25, 0.25, 0.25, -0.25]
    plt.plot(table_x, table_y, 'k-', linewidth=2, label='Table Edge')
    
    # Draw randomization area
    rand_x = [-0.15, 0.15, 0.15, -0.15, -0.15]
    rand_y = [-0.15, -0.15, 0.15, 0.15, -0.15]
    plt.fill(rand_x, rand_y, alpha=0.2, color='blue', label='Starting Area')
    
    plt.scatter([0], [0], color='red', s=100, marker='x', label='Center Target')
    
    plt.xlim(-0.3, 0.3)
    plt.ylim(-0.3, 0.3)
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.title('Ball Starting Positions in RL Training\n(100 random samples)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    
    plt.tight_layout()
    plt.show()
    
    print(f"Starting positions cover {0.3*0.3:.2f} square units")
    print(f"Table area is {0.5*0.5:.2f} square units")
    print(f"Coverage: {(0.3*0.3)/(0.5*0.5)*100:.1f}% of table area")


def analyze_action_space():
    """Analyze the action space"""
    env = BallBalanceEnv(render_mode="rgb_array")
    
    print(f"\nAction Space Analysis:")
    print(f"Action space: {env.action_space}")
    print(f"Action low: {env.action_space.low}")
    print(f"Action high: {env.action_space.high}")
    print(f"Max angle change per step: ±{env.action_space.high[0]:.3f} radians (±{np.degrees(env.action_space.high[0]):.1f}°)")
    print(f"Max total angle: ±0.1 radians (±{np.degrees(0.1):.1f}°)")
    
    env.close()


def convergence_tips():
    """Print tips for better convergence"""
    print("\n" + "="*60)
    print("🔧 CONVERGENCE TROUBLESHOOTING TIPS:")
    print("="*60)
    
    print("\n1. 📊 REWARD FUNCTION ISSUES:")
    print("   - Rewards too sparse (only at center)")
    print("   - Huge penalty spikes (-100) cause instability")
    print("   - Solution: Smooth, shaped rewards ✅")
    
    print("\n2. 🎛️ HYPERPARAMETER ISSUES:")
    print("   - Learning rate too high (unstable)")
    print("   - Batch size too large (slow learning)")
    print("   - Solution: Smaller, more frequent updates ✅")
    
    print("\n3. 📏 ENVIRONMENT ISSUES:")
    print("   - Action space too small/large")
    print("   - Episodes too short/long")
    print("   - Observation scaling problems")
    
    print("\n4. 🎯 TRAINING ISSUES:")
    print("   - Not enough exploration")
    print("   - Premature convergence to bad policy")
    print("   - Need curriculum learning")
    
    print("\n5. 📈 WHAT TO MONITOR:")
    print("   - eval/mean_reward should trend upward")
    print("   - eval/mean_ep_length should increase")
    print("   - Policy loss should decrease but not to zero")
    print("   - Entropy should stay > 0 (exploration)")
    
    print("\n6. 🚨 RED FLAGS:")
    print("   - Rewards stuck at same value")
    print("   - Episode length never improves")
    print("   - Huge spikes in loss")
    print("   - Entropy drops to near zero too quickly")


def test_pid_baseline():
    """Test what reward your PID controller would get"""
    print("\nPID Baseline Test:")
    print("Run: python compare_control.py --control pid")
    print("Let it run for a while and observe typical performance")
    print("This gives you a target for RL to beat!")


def main():
    print("🤖 Ball Balance RL Debugging Tool")
    print("="*50)
    
    # Run all tests
    test_reward_function()
    analyze_action_space()
    
    print("\nDo you want to see starting position visualization? (y/n): ", end="")
    if input().lower().startswith('y'):
        visualize_starting_positions()
    
    print("\nDo you want to test random policy? (y/n): ", end="")
    if input().lower().startswith('y'):
        test_random_policy(3)
    
    test_pid_baseline()
    convergence_tips()
    
    print(f"\n🎯 NEXT STEPS:")
    print(f"1. Try the improved reward function (already applied)")
    print(f"2. Train with better hyperparameters: python train_rl.py --mode train")
    print(f"3. Monitor with TensorBoard: tensorboard --logdir=./tensorboard_logs/")
    print(f"4. Look for smoother, upward trends in rewards")


if __name__ == "__main__":
    main()
