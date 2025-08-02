#!/usr/bin/env python3
"""Quick test to see raw IMU readings"""

import sys
import time
sys.path.append('imu')

from imu_simple import SimpleBNO055Interface

def main():
    print("🧭 Quick Raw IMU Reading Test")
    print("=" * 40)
    
    imu = SimpleBNO055Interface("COM8")
    if not imu.connect():
        print("❌ Failed to connect")
        return
    
    print("📊 Raw IMU readings (10 samples):")
    print("Time   | Heading | Pitch  | Roll   ")
    print("-" * 35)
    
    for i in range(10):
        line = imu.read_line()
        if line and line.startswith("DATA: "):
            try:
                data_part = line[6:]
                heading, pitch, roll = map(float, data_part.split(','))
                print(f"{i+1:2d}     | {heading:6.1f}° | {pitch:+6.1f}° | {roll:+6.1f}°")
            except:
                print(f"{i+1:2d}     | Invalid data: {line}")
        else:
            print(f"{i+1:2d}     | No data received")
        time.sleep(0.5)
    
    imu.cleanup()
    print("\n💡 These are the RAW readings from your IMU")
    print("   Your calibration file has offsets to make these read ~0° when level")

if __name__ == "__main__":
    main()
