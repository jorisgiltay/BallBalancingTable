#!/usr/bin/env python3
"""
Comprehensive IMU Diagnostic Tool

This tool performs deep diagnostics on the BNO055 to understand
why it might be reading exactly 0.0 for all angles.
"""

import sys
import os
import time
import serial
import numpy as np

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append('.')

class IMUDiagnostics:
    def __init__(self, port="COM8", baudrate=115200):
        self.port = port
        self.baudrate = baudrate
        self.ser = None
    
    def connect(self):
        """Connect to IMU"""
        try:
            print(f"🔌 Connecting to IMU on {self.port} at {self.baudrate} baud...")
            self.ser = serial.Serial(self.port, self.baudrate, timeout=2)
            time.sleep(2)  # Wait for Arduino reset
            
            # Clear any buffered data
            self.ser.reset_input_buffer()
            print("✅ Connected successfully")
            return True
        except Exception as e:
            print(f"❌ Connection failed: {e}")
            return False
    
    def read_raw_data(self, duration=10):
        """Read raw data from IMU for analysis"""
        print(f"\n📡 Reading raw IMU data for {duration} seconds...")
        print("Raw Arduino Output:")
        print("-" * 50)
        
        start_time = time.time()
        all_lines = []
        
        while time.time() - start_time < duration:
            try:
                if self.ser.in_waiting > 0:
                    line = self.ser.readline().decode('utf-8').strip()
                    if line:
                        print(f"  {line}")
                        all_lines.append(line)
            except Exception as e:
                print(f"Read error: {e}")
                break
        
        return all_lines
    
    def analyze_data_patterns(self, lines):
        """Analyze patterns in the data"""
        print(f"\n🔍 Data Pattern Analysis:")
        print("=" * 50)
        
        data_lines = [l for l in lines if l.startswith("DATA:")]
        gyro_lines = [l for l in lines if l.startswith("GYRO:")]
        error_lines = [l for l in lines if "ERROR" in l or "WARNING" in l]
        
        print(f"📊 Data Summary:")
        print(f"   • Total lines: {len(lines)}")
        print(f"   • DATA lines: {len(data_lines)}")
        print(f"   • GYRO lines: {len(gyro_lines)}")
        print(f"   • Error/Warning lines: {len(error_lines)}")
        
        if error_lines:
            print(f"\n⚠️ Errors/Warnings found:")
            for line in error_lines:
                print(f"   {line}")
        
        # Analyze DATA lines
        if data_lines:
            print(f"\n📐 Euler Angle Analysis:")
            unique_data = set(data_lines)
            print(f"   • Unique DATA patterns: {len(unique_data)}")
            
            if len(unique_data) <= 5:
                print("   • All unique patterns:")
                for pattern in unique_data:
                    count = data_lines.count(pattern)
                    print(f"     '{pattern}' appears {count} times")
            
            # Parse numeric values
            euler_values = []
            for line in data_lines:
                try:
                    data_part = line[5:].strip()  # Remove "DATA:"
                    heading, pitch, roll = map(float, data_part.split(','))
                    euler_values.append((heading, pitch, roll))
                except:
                    continue
            
            if euler_values:
                headings = [v[0] for v in euler_values]
                pitches = [v[1] for v in euler_values]
                rolls = [v[2] for v in euler_values]
                
                print(f"   • Heading range: {min(headings):.1f}° to {max(headings):.1f}°")
                print(f"   • Pitch range: {min(pitches):.1f}° to {max(pitches):.1f}°")
                print(f"   • Roll range: {min(rolls):.1f}° to {max(rolls):.1f}°")
                
                # Check for exactly zero values
                zero_headings = sum(1 for h in headings if h == 0.0)
                zero_pitches = sum(1 for p in pitches if p == 0.0)
                zero_rolls = sum(1 for r in rolls if r == 0.0)
                
                print(f"   • Exactly zero readings:")
                print(f"     - Heading: {zero_headings}/{len(headings)} ({100*zero_headings/len(headings):.1f}%)")
                print(f"     - Pitch: {zero_pitches}/{len(pitches)} ({100*zero_pitches/len(pitches):.1f}%)")
                print(f"     - Roll: {zero_rolls}/{len(rolls)} ({100*zero_rolls/len(rolls):.1f}%)")
        
        # Analyze GYRO lines
        if gyro_lines:
            print(f"\n🌀 Gyroscope Analysis:")
            unique_gyro = set(gyro_lines)
            print(f"   • Unique GYRO patterns: {len(unique_gyro)}")
            
            if len(unique_gyro) <= 5:
                print("   • All unique patterns:")
                for pattern in unique_gyro:
                    count = gyro_lines.count(pattern)
                    print(f"     '{pattern}' appears {count} times")
            
            # Parse gyro values
            gyro_values = []
            for line in gyro_lines:
                try:
                    data_part = line[5:].strip()  # Remove "GYRO:"
                    gx, gy, gz = map(float, data_part.split(','))
                    gyro_values.append((gx, gy, gz))
                except:
                    continue
            
            if gyro_values:
                gx_vals = [v[0] for v in gyro_values]
                gy_vals = [v[1] for v in gyro_values]
                gz_vals = [v[2] for v in gyro_values]
                
                print(f"   • X-gyro range: {min(gx_vals):.3f} to {max(gx_vals):.3f} rad/s")
                print(f"   • Y-gyro range: {min(gy_vals):.3f} to {max(gy_vals):.3f} rad/s")
                print(f"   • Z-gyro range: {min(gz_vals):.3f} to {max(gz_vals):.3f} rad/s")
    
    def send_diagnostic_commands(self):
        """Send diagnostic commands to Arduino (if supported)"""
        print(f"\n🔧 Sending diagnostic commands...")
        
        # Try some common diagnostic commands
        commands = ["?", "STATUS", "CAL", "INFO"]
        
        for cmd in commands:
            print(f"   Sending: '{cmd}'")
            self.ser.write((cmd + '\n').encode())
            time.sleep(0.5)
            
            # Read response
            response_lines = []
            start_time = time.time()
            while time.time() - start_time < 2:
                try:
                    if self.ser.in_waiting > 0:
                        line = self.ser.readline().decode('utf-8').strip()
                        if line and not line.startswith("DATA:") and not line.startswith("GYRO:"):
                            response_lines.append(line)
                            print(f"     Response: {line}")
                except:
                    break
            
            if not response_lines:
                print(f"     No response to '{cmd}'")
    
    def physical_movement_test(self):
        """Test if IMU responds to physical movement"""
        print(f"\n🏃 Physical Movement Test")
        print("=" * 50)
        print("We'll monitor the IMU while you move it to see if it responds...")
        print()
        
        # Baseline reading
        print("1. BASELINE - Keep IMU completely still:")
        input("   Press Enter when ready...")
        baseline_data = self.read_movement_data(5, "STILL")
        
        print("\n2. TILT TEST - Gently tilt the IMU left and right:")
        input("   Press Enter when ready...")
        tilt_data = self.read_movement_data(10, "TILTING")
        
        print("\n3. ROTATION TEST - Slowly rotate the IMU:")
        input("   Press Enter when ready...")
        rotate_data = self.read_movement_data(10, "ROTATING")
        
        # Analyze responsiveness
        self.analyze_movement_response(baseline_data, tilt_data, rotate_data)
    
    def read_movement_data(self, duration, phase):
        """Read data during movement test"""
        print(f"   📊 Recording {phase} data for {duration} seconds...")
        
        start_time = time.time()
        euler_data = []
        gyro_data = []
        
        while time.time() - start_time < duration:
            try:
                if self.ser.in_waiting > 0:
                    line = self.ser.readline().decode('utf-8').strip()
                    
                    if line.startswith("DATA:"):
                        try:
                            data_part = line[5:].strip()
                            h, p, r = map(float, data_part.split(','))
                            euler_data.append((h, p, r))
                        except:
                            pass
                    
                    elif line.startswith("GYRO:"):
                        try:
                            data_part = line[5:].strip()
                            gx, gy, gz = map(float, data_part.split(','))
                            gyro_data.append((gx, gy, gz))
                        except:
                            pass
            except:
                break
        
        print(f"   ✅ Recorded {len(euler_data)} Euler samples, {len(gyro_data)} gyro samples")
        return {"euler": euler_data, "gyro": gyro_data, "phase": phase}
    
    def analyze_movement_response(self, baseline, tilt, rotate):
        """Analyze if IMU responds to movement"""
        print(f"\n🎯 Movement Response Analysis:")
        print("=" * 50)
        
        for data_set in [baseline, tilt, rotate]:
            phase = data_set["phase"]
            euler_data = data_set["euler"]
            gyro_data = data_set["gyro"]
            
            print(f"\n📊 {phase} Phase:")
            
            if euler_data:
                headings = [d[0] for d in euler_data]
                pitches = [d[1] for d in euler_data]
                rolls = [d[2] for d in euler_data]
                
                h_range = max(headings) - min(headings)
                p_range = max(pitches) - min(pitches)
                r_range = max(rolls) - min(rolls)
                
                print(f"   📐 Euler Angles:")
                print(f"     • Heading variation: {h_range:.2f}° (min: {min(headings):.1f}°, max: {max(headings):.1f}°)")
                print(f"     • Pitch variation: {p_range:.2f}° (min: {min(pitches):.1f}°, max: {max(pitches):.1f}°)")
                print(f"     • Roll variation: {r_range:.2f}° (min: {min(rolls):.1f}°, max: {max(rolls):.1f}°)")
                
                # Check if all exactly zero
                all_zero = all(h == 0.0 and p == 0.0 and r == 0.0 for h, p, r in euler_data)
                if all_zero:
                    print(f"     ⚠️ ALL EULER ANGLES ARE EXACTLY ZERO!")
                else:
                    print(f"     ✅ Euler angles show variation")
            
            if gyro_data:
                gx_vals = [d[0] for d in gyro_data]
                gy_vals = [d[1] for d in gyro_data]
                gz_vals = [d[2] for d in gyro_data]
                
                gx_range = max(gx_vals) - min(gx_vals)
                gy_range = max(gy_vals) - min(gy_vals)
                gz_range = max(gz_vals) - min(gz_vals)
                
                print(f"   🌀 Gyroscope:")
                print(f"     • X-axis variation: {gx_range:.3f} rad/s")
                print(f"     • Y-axis variation: {gy_range:.3f} rad/s")
                print(f"     • Z-axis variation: {gz_range:.3f} rad/s")
                
                # Check if gyro is responsive
                total_gyro_activity = gx_range + gy_range + gz_range
                if total_gyro_activity < 0.01:
                    print(f"     ⚠️ Very little gyroscope activity!")
                else:
                    print(f"     ✅ Gyroscope shows movement response")
    
    def diagnose_bno055_state(self):
        """Try to diagnose BNO055 internal state issues"""
        print(f"\n🔬 BNO055 State Diagnosis")
        print("=" * 50)
        print("The BNO055 has internal calibration states that can get reset.")
        print("If all Euler angles are exactly 0.0, it might indicate:")
        print("  • Internal calibration lost")
        print("  • Sensor fusion not working")
        print("  • Hardware issue")
        print("  • Magnetic interference")
        print()
        
        # Look for calibration status if Arduino supports it
        print("💡 Recommendations:")
        print("  1. Power cycle the Arduino/IMU completely")
        print("  2. Try moving the IMU in a figure-8 pattern for 30 seconds")
        print("  3. Keep IMU away from magnetic interference (motors, metal)")
        print("  4. Check if Arduino code has calibration status output")
        print("  5. Consider reflashing Arduino with calibration status code")
    
    def cleanup(self):
        """Clean up connection"""
        if self.ser:
            self.ser.close()
    
    def run_full_diagnosis(self):
        """Run complete IMU diagnosis"""
        print("🩺 Comprehensive IMU Diagnostics")
        print("=" * 60)
        print()
        
        if not self.connect():
            return
        
        try:
            # Read raw data
            raw_lines = self.read_raw_data(10)
            
            # Analyze patterns
            self.analyze_data_patterns(raw_lines)
            
            # Try diagnostic commands
            self.send_diagnostic_commands()
            
            # Physical movement test
            self.physical_movement_test()
            
            # BNO055 state diagnosis
            self.diagnose_bno055_state()
            
            print(f"\n🏁 Diagnosis Complete!")
            print("=" * 60)
            
        except KeyboardInterrupt:
            print("\n🛑 Diagnosis interrupted by user")
        finally:
            self.cleanup()

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Comprehensive IMU Diagnostics")
    parser.add_argument("--port", default="COM8", help="IMU COM port")
    parser.add_argument("--baudrate", type=int, default=115200, help="Serial baudrate")
    
    args = parser.parse_args()
    
    diagnostics = IMUDiagnostics(args.port, args.baudrate)
    diagnostics.run_full_diagnosis()

if __name__ == "__main__":
    main()
