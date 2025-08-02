#!/usr/bin/env python3
"""
Test IMU Control Mode

This script demonstrates the new IMU control mode where the simulation table
follows your real IMU movements.
"""

def test_imu_control_help():
    """Show help for the new IMU control mode"""
    print("🎮 IMU Control Mode - NEW FEATURE!")
    print("=" * 50)
    print()
    print("🎯 What it does:")
    print("   • Simulation table follows your real IMU movements")
    print("   • Physically tilt your table → simulation responds instantly")
    print("   • Perfect for testing IMU accuracy and ball physics")
    print("   • Automatic offset calibration handles IMU baseline errors")
    print()
    print("🔧 How to use:")
    print("   1. Connect your BNO055 IMU via Arduino")
    print("   2. Make sure your table is LEVEL")
    print("   3. Run: python compare_control.py --imu-control --imu-port COM7")
    print("   4. Press 'c' in simulation to calibrate offsets")
    print("   5. Tilt your physical table and watch simulation follow!")
    print()
    print("📊 Command examples:")
    print("   # Basic IMU control:")
    print("   python compare_control.py --imu-control")
    print()
    print("   # IMU control + visuals:")
    print("   python compare_control.py --imu-control --visuals")
    print()
    print("   # IMU control + servos (full hardware):")
    print("   python compare_control.py --imu-control --servos")
    print()
    print("   # Different COM port:")
    print("   python compare_control.py --imu-control --imu-port COM7")
    print()
    print("🧭 Calibration process:")
    print("   1. Start with table perfectly level")
    print("   2. Press 'c' key in simulation")
    print("   3. Keep table stable for 5 seconds")
    print("   4. Offsets calculated automatically")
    print("   5. Now tilt table - simulation follows with zero baseline error!")
    print()
    print("💡 Benefits:")
    print("   • Handles IMU offset/bias automatically")
    print("   • Real-time validation of IMU accuracy")
    print("   • Test ball physics with real table movements")
    print("   • Great for debugging IMU integration")
    print()
    print("⚙️ Technical details:")
    print("   • 80% scaling factor applied (configurable)")
    print("   • Automatic offset subtraction")
    print("   • Thread-safe IMU reading at 100Hz")
    print("   • Real-time calibration quality assessment")

if __name__ == "__main__":
    test_imu_control_help()
