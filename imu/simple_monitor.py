#!/usr/bin/env python3
"""
Simple IMU Monitor - Just shows the values you want
"""

import serial
import time
import os

def load_calibration():
    """Load calibration offsets"""
    try:
        pitch_offset = 0.0
        roll_offset = 0.0
        
        cal_file = "embedded_imu_calibration.txt"
        if os.path.exists(cal_file):
            with open(cal_file, 'r') as f:
                for line in f:
                    if line.startswith('level_pitch_offset'):
                        pitch_offset = float(line.split('=')[1].strip())
                    elif line.startswith('level_roll_offset'):
                        roll_offset = float(line.split('=')[1].strip())
        
        return pitch_offset, roll_offset
    except:
        return 0.0, 0.0

def monitor_imu(port="COM8"):
    """Monitor IMU with simple text output"""
    
    # Load calibration
    pitch_offset, roll_offset = load_calibration()
    print(f"Loaded calibration: Pitch offset = {pitch_offset:+.2f}°, Roll offset = {roll_offset:+.2f}°")
    
    # Connect to IMU
    try:
        ser = serial.Serial(port, 115200, timeout=1)
        time.sleep(2)  # Wait for Arduino startup
        print(f"Connected to {port}")
        print()
        print("Time      | Sample | Raw Pitch | Raw Roll | Cal Pitch | Cal Roll")
        print("----------|--------|-----------|----------|-----------|----------")
        
        sample_count = 0
        start_time = time.time()
        
        while True:
            try:
                if ser.in_waiting > 0:
                    line = ser.readline().decode('utf-8').strip()
                    
                    if line.startswith("DATA:"):
                        # Skip startup zeros
                        data_part = line[5:].strip()
                        heading, pitch, roll = map(float, data_part.split(','))
                        
                        if not (heading == 0.0 and pitch == 0.0 and roll == 0.0):
                            sample_count += 1
                            elapsed = time.time() - start_time
                            
                            # Calculate calibrated values
                            cal_pitch = pitch + pitch_offset
                            cal_roll = roll + roll_offset
                            
                            # Print the values
                            print(f"{elapsed:8.1f}s | {sample_count:6d} | {pitch:+8.1f}° | {roll:+7.1f}° | {cal_pitch:+8.1f}° | {cal_roll:+7.1f}°")
                            
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Error: {e}")
                break
        
        ser.close()
        print("\nDisconnected")
        
    except Exception as e:
        print(f"Failed to connect to {port}: {e}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", default="COM8", help="COM port")
    args = parser.parse_args()
    
    monitor_imu(args.port)
