#!/usr/bin/env python3
"""
Test script for IMU integration with compare_control
"""

import sys
import time

def test_imu_import():
    """Test if IMU integration imports work"""
    print("🧪 Testing IMU integration imports...")
    
    try:
        # Test sys.path modification
        sys.path.append('imu')
        print("✅ sys.path.append('imu') - OK")
        
        # Test IMU import
        from imu_simple import SimpleBNO055Interface
        print("✅ IMU import - OK")
        
        # Test IMU initialization
        imu = SimpleBNO055Interface(port='COM6')
        print("✅ IMU initialization - OK")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Other error: {e}")
        return False

def test_compare_control_import():
    """Test if compare_control imports with IMU support"""
    print("🧪 Testing compare_control with IMU integration...")
    
    try:
        from compare_control import BallBalanceComparison
        print("✅ compare_control import - OK")
        
        # Test initialization with IMU disabled (safe test)
        simulator = BallBalanceComparison(
            control_method="pid",
            control_freq=50,
            enable_visuals=False,
            enable_servos=False,
            camera_mode="simulation",
            enable_imu=False,  # Disabled for testing
            imu_port="COM6"
        )
        print("✅ BallBalanceComparison initialization with IMU support - OK")
        
        # Test IMU feedback method
        feedback = simulator.get_imu_feedback()
        print(f"✅ IMU feedback method - OK: {feedback}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function"""
    print("🚀 Testing IMU Integration with Ball Balancing Control")
    print("=" * 60)
    
    # Test 1: IMU imports
    if not test_imu_import():
        print("❌ IMU import test failed - check imu_simple.py exists")
        return False
    
    print()
    
    # Test 2: compare_control with IMU
    if not test_compare_control_import():
        print("❌ compare_control IMU integration test failed")
        return False
    
    print()
    print("✅ All tests passed! IMU integration is ready.")
    print()
    print("🔧 Usage examples:")
    print("   # Run with IMU feedback enabled:")
    print("   python compare_control.py --imu --imu-port COM6")
    print()
    print("   # Run with IMU + visuals + servos:")
    print("   python compare_control.py --imu --visuals --servos")
    print()
    print("   # Run with IMU + camera (full hardware mode):")
    print("   python compare_control.py --imu --camera real --servos")
    
    return True

if __name__ == "__main__":
    main()
