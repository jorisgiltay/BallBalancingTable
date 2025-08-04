#!/usr/bin/env python3
"""
Setup script for Ball Balancing RL project
"""

import os
import subprocess
import sys


def install_requirements():
    """Install required packages"""
    print("Installing required packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✓ All packages installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Error installing packages: {e}")
        return False


def check_project_structure():
    """Check if the project structure is correct"""
    print("Checking project structure...")
    
    required_files = [
        "compare_control.py",
        "requirements.txt",
        "reinforcement_learning/train_rl.py",
        "reinforcement_learning/ball_balance_env.py"
    ]
    
    missing_files = []
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✓ Found: {file_path}")
        else:
            print(f"✗ Missing: {file_path}")
            missing_files.append(file_path)
    
    if missing_files:
        print(f"\n⚠️  Warning: {len(missing_files)} required files are missing.")
        print("Please ensure you have the complete project structure.")
        return False
    
    print("✓ Project structure looks good!")
    return True


def create_directories():
    """Create necessary directories"""
    # Main project directories
    main_directories = ["models", "tensorboard_logs", "checkpoints"]
    
    # Reinforcement learning specific directories
    rl_directories = [
        "reinforcement_learning/models", 
        "reinforcement_learning/tensorboard_logs", 
        "reinforcement_learning/checkpoints"
    ]
    
    print("Creating main project directories...")
    for directory in main_directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✓ Created directory: {directory}")
    
    print("Creating reinforcement learning directories...")
    for directory in rl_directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✓ Created directory: {directory}")


def main():
    print("=== Ball Balancing RL Project Setup ===")
    print()
    
    # Check project structure
    if not check_project_structure():
        print("\n⚠️  Project structure check failed. Setup may be incomplete.")
        print("Continuing anyway...")
    print()
    
    # Install requirements
    if not install_requirements():
        print("Setup failed. Please check your Python environment.")
        sys.exit(1)
    
    # Create directories
    create_directories()
    
    print()
    print("=== Setup Complete! ===")
    print()
    print("Quick Start Guide:")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print()
    print("🎮 PID Control (Main Directory):")
    print("   python compare_control.py --control pid")
    print("   python compare_control.py --control pid --visuals")
    print("   python compare_control.py --control pid --servos")
    print()
    print("🤖 Reinforcement Learning (RL Directory):")
    print("   cd reinforcement_learning")
    print("   python train_rl.py --mode train --tensorboard")
    print("   python train_rl.py --mode test")
    print("   cd ..")
    print()
    print("🔄 Compare Both (Main Directory):")
    print("   python compare_control.py --control rl")
    print()
    print("📊 Monitor Training:")
    print("   Use --tensorboard flag during training")
    print("   Or manually: tensorboard --logdir=reinforcement_learning/tensorboard_logs/")
    print()
    print("🎯 Hardware Integration:")
    print("   --servos     : Enable servo control")
    print("   --camera     : Use camera input (hybrid/real)")
    print("   --imu        : Enable IMU feedback")
    print("   --visuals    : Show real-time dashboard")
    print()
    print("⌨️  Simulation Controls:")
    print("   'r' - Reset ball position")
    print("   'p' - Switch to PID control")
    print("   'l' - Switch to RL control")
    print("   'q' - Quit simulation")


if __name__ == "__main__":
    main()
