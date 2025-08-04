#!/usr/bin/env python3
"""
Test IMU-Servo Alignment

This script helps verify that the IMU coordinate system matches
the servo coordinate system for proper feedback control.
"""

import sys
import time
import math

# Import dependencies with error handling
try:
    sys.path.append('imu')
    from imu_simple import SimpleBNO055Interface
    IMU_AVAILABLE = True
except ImportError:
    IMU_AVAILABLE = False
    print("⚠️ IMU interface not available")

try:
    from servo.servo_controller import ServoController
    SERVO_AVAILABLE = True
except ImportError:
    SERVO_AVAILABLE = False
    print("⚠️ Servo controller not available")

def test_alignment():
    """Test IMU and servo coordinate alignment"""
    print("🧪 IMU-Servo Alignment Test")
    print("=" * 50)
    
    # Initialize IMU
    imu = None
    if IMU_AVAILABLE:
        imu = SimpleBNO055Interface("COM6")  # Update port as needed
        if imu.connect():
            print("✅ IMU connected")
        else:
            print("❌ IMU connection failed")
            imu = None
    
    # Initialize servos
    servo = None
    if SERVO_AVAILABLE:
        servo = ServoController()
        if servo.connect():
            print("✅ Servos connected")
        else:
            print("❌ Servo connection failed")
            servo = None
    
    if not imu and not servo:
        print("❌ No hardware available for testing")
        return
    
    print("\n📋 Test Procedure:")
    print("1. Baseline reading (level table)")
    print("2. Pitch test (tilt forward/backward)")  
    print("3. Roll test (tilt left/right)")
    print("4. Return to center")
    
    try:
        # Test sequence
        test_angles = [
            (0, 0, "Baseline - Level table"),
            (2, 0, "Pitch +2° (servo ID 1 positive)"),
            (-2, 0, "Pitch -2° (servo ID 1 negative)"),
            (0, 2, "Roll +2° (servo ID 2 positive)"),
            (0, -2, "Roll -2° (servo ID 2 negative)"),
            (0, 0, "Return to center")
        ]
        
        for pitch_deg, roll_deg, description in test_angles:
            print(f"\n🎯 {description}")
            
            # Move servos if available
            if servo:
                print(f"   Moving servos: Pitch={pitch_deg}°, Roll={roll_deg}°")
                servo.set_table_angles_degrees(pitch_deg, roll_deg)
                time.sleep(2)  # Allow movement to complete
            
            # Read IMU if available
            if imu:
                print("   Reading IMU...")
                imu_readings = []
                for _ in range(10):  # Take multiple readings
                    line = imu.read_line()
                    if line and line.startswith("DATA: "):
                        try:
                            data_part = line[6:]
                            heading, pitch_imu, roll_imu = map(float, data_part.split(','))
                            imu_readings.append((heading, pitch_imu, roll_imu))
                        except:
                            pass
                    time.sleep(0.1)
                
                if imu_readings:
                    # Average the readings
                    avg_heading = sum(r[0] for r in imu_readings) / len(imu_readings)
                    avg_pitch = sum(r[1] for r in imu_readings) / len(imu_readings)
                    avg_roll = sum(r[2] for r in imu_readings) / len(imu_readings)
                    
                    print(f"   IMU Average: H={avg_heading:+.1f}°, P={avg_pitch:+.1f}°, R={avg_roll:+.1f}°")
                    
                    # Check alignment
                    if servo:
                        pitch_error = pitch_deg - avg_pitch
                        roll_error = roll_deg - avg_roll
                        print(f"   Alignment Error: Pitch={pitch_error:+.1f}°, Roll={roll_error:+.1f}°")
                        
                        # Warn about large errors
                        if abs(pitch_error) > 1.0:
                            print(f"   ⚠️ Large pitch error - coordinate system may be misaligned!")
                        if abs(roll_error) > 1.0:
                            print(f"   ⚠️ Large roll error - coordinate system may be misaligned!")
                else:
                    print("   ❌ No valid IMU readings received")
            
            # Pause between tests
            if description != "Return to center":
                input("   Press Enter to continue...")
        
        print("\n✅ Alignment test completed!")
        print("\n📊 Analysis Guide:")
        print("• IMU pitch should match servo pitch command")
        print("• IMU roll should match servo roll command") 
        print("• Large errors (>1°) indicate coordinate misalignment")
        print("• Sign errors indicate axis inversion")
        
    except KeyboardInterrupt:
        print("\n🛑 Test interrupted")
    
    finally:
        # Cleanup
        if servo:
            servo.set_table_angles_degrees(0, 0)  # Return to center
            servo.disconnect()
        if imu:
            imu.cleanup()

if __name__ == "__main__":
    test_alignment()
