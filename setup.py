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


def create_directories():
    """Create necessary directories"""
    directories = ["models", "tensorboard_logs"]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✓ Created directory: {directory}")


def main():
    print("=== Ball Balancing RL Project Setup ===")
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
    print("1. Test PID control:     python compare_control.py --control pid")
    print("2. Train RL agent:       python train_rl.py --mode train")
    print("3. Test RL agent:        python train_rl.py --mode test")
    print("4. Compare both:         python compare_control.py --control rl")
    print()
    print("During simulation, use these keys:")
    print("  'r' - Reset ball position")
    print("  'p' - Switch to PID control")
    print("  'l' - Switch to RL control")
    print("  'q' - Quit simulation")


if __name__ == "__main__":
    main()
