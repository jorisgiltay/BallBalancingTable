#!/usr/bin/env python3
"""Debug IMU data format"""

import sys
import time
sys.path.append('imu')

from imu_simple import SimpleBNO055Interface

def main():
    print("🔍 IMU Data Format Debug")
    print("=" * 40)
    
    imu = SimpleBNO055Interface("COM8")
    if not imu.connect():
        print("❌ Failed to connect")
        return
    
    print("📊 Raw lines from IMU (first 20 lines):")
    print("-" * 60)
    
    for i in range(20):
        line = imu.read_line()
        if line:
            print(f"{i+1:2d}: '{line}' (len={len(line)})")
            
            # Test different parsing methods
            if "DATA" in line:
                print(f"    -> Contains 'DATA'")
                if line.startswith("DATA:"):
                    print(f"    -> Starts with 'DATA:' (no space)")
                    try:
                        data_part = line[5:]
                        print(f"    -> After removing 'DATA:': '{data_part}'")
                        values = data_part.split(',')
                        print(f"    -> Split values: {values}")
                        if len(values) == 3:
                            h, p, r = map(float, values)
                            print(f"    -> Parsed: H={h}, P={p}, R={r}")
                    except Exception as e:
                        print(f"    -> Parse error: {e}")
                        
                if line.startswith("DATA: "):
                    print(f"    -> Starts with 'DATA: ' (with space)")
                    try:
                        data_part = line[6:]
                        print(f"    -> After removing 'DATA: ': '{data_part}'")
                        values = data_part.split(',')
                        print(f"    -> Split values: {values}")
                        if len(values) == 3:
                            h, p, r = map(float, values)
                            print(f"    -> Parsed: H={h}, P={p}, R={r}")
                    except Exception as e:
                        print(f"    -> Parse error: {e}")
        else:
            print(f"{i+1:2d}: No data")
        time.sleep(0.2)
    
    imu.cleanup()

if __name__ == "__main__":
    main()
