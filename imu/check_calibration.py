#!/usr/bin/env python3
"""
Quick IMU Calibration Check Tool

This script quickly verifies if your IMU calibration is still accurate
without requiring a full recalibration process.
"""

import sys
import os
import time
import numpy as np

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append('.')

try:
    from imu_simple import SimpleBNO055Interface
except ImportError:
    print("❌ Could not import imu_simple. Make sure you're in the right directory.")
    sys.exit(1)

class CalibrationChecker:
    def __init__(self, port="COM7"):
        self.port = port
        self.imu_interface = None
        self.pitch_offset = 0.0
        self.roll_offset = 0.0
        self.calibration_loaded = False
    
    def load_existing_calibration(self):
        """Load existing calibration data"""
        try:
            # Try multiple possible locations
            possible_files = [
                "embedded_imu_calibration.txt",  # If running from imu/ folder
                "../embedded_imu_calibration.txt",  # If running from main folder and file is in root
                "imu/embedded_imu_calibration.txt"  # If running from main folder and file is in imu/
            ]
            
            filename = None
            for f in possible_files:
                if os.path.exists(f):
                    filename = f
                    break
            
            if not filename:
                print(f"❌ No calibration file found. Tried:")
                for f in possible_files:
                    print(f"   - {f}")
                return False
            
            print(f"📁 Loading calibration from {filename}...")
            
            with open(filename, 'r') as f:
                lines = f.readlines()
            
            for line in lines:
                line = line.strip()
                if line.startswith('level_pitch_offset'):
                    self.pitch_offset = float(line.split('=')[1].strip())
                elif line.startswith('level_roll_offset'):
                    self.roll_offset = float(line.split('=')[1].strip())
            
            if self.pitch_offset != 0.0 or self.roll_offset != 0.0:
                self.calibration_loaded = True
                print(f"✅ Loaded calibration offsets:")
                print(f"   📐 Pitch offset: {self.pitch_offset:+.2f}°")
                print(f"   📐 Roll offset:  {self.roll_offset:+.2f}°")
                return True
            else:
                print(f"⚠️ Calibration file found but offsets are zero")
                return False
                
        except Exception as e:
            print(f"❌ Failed to load calibration: {e}")
            return False
    
    def connect_imu(self):
        """Connect to IMU"""
        print(f"🧭 Connecting to IMU on {self.port}...")
        self.imu_interface = SimpleBNO055Interface(self.port)
        
        if self.imu_interface.connect():
            print("✅ IMU connected successfully")
            return True
        else:
            print("❌ Failed to connect to IMU")
            return False
    
    def get_imu_reading(self):
        """Get a single IMU reading, waiting for real values (not startup zeros)"""
        if not self.imu_interface:
            return None
        
        # Try to get a reading for up to 10 seconds, skipping startup zeros
        for _ in range(100):  # 100 attempts at 0.1s intervals
            line = self.imu_interface.read_line()
            if line and line.startswith("DATA:"):
                try:
                    data_part = line[5:].strip()  # Remove "DATA:" prefix and whitespace
                    heading, pitch, roll = map(float, data_part.split(','))
                    
                    # Skip startup zeros - wait for real IMU values
                    if not (heading == 0.0 and pitch == 0.0 and roll == 0.0):
                        return heading, pitch, roll
                        
                except:
                    continue
            time.sleep(0.1)
        
        return None
    
    def check_calibration_accuracy(self, samples=20, tolerance_degrees=1.0):
        """Check if current calibration is still accurate"""
        print(f"\n🎯 Calibration Accuracy Check")
        print("=" * 50)
        
        if not self.calibration_loaded:
            print("❌ No calibration loaded - cannot check accuracy")
            return False
        
        print(f"📋 Instructions:")
        print(f"   1. Place your table on a LEVEL surface")
        print(f"   2. Make sure the table is not tilted in any direction")
        print(f"   3. Keep the setup stable during measurement")
        print()
        
        input("Press Enter when table is level and stable...")
        
        print(f"📊 Taking {samples} measurements to check calibration...")
        
        pitch_readings = []
        roll_readings = []
        
        for i in range(samples):
            reading = self.get_imu_reading()
            if reading is None:
                print(f"❌ Failed to get IMU reading {i+1}")
                continue
            
            heading, pitch, roll = reading
            
            # Apply existing calibration
            calibrated_pitch = pitch + self.pitch_offset
            calibrated_roll = roll + self.roll_offset
            
            pitch_readings.append(calibrated_pitch)
            roll_readings.append(calibrated_roll)
            
            if (i + 1) % 5 == 0:
                print(f"   Sample {i + 1:2d}/{samples}: P{calibrated_pitch:+.1f}° R{calibrated_roll:+.1f}° (calibrated)")
                print(f"                        Raw: P{pitch:+.1f}° R{roll:+.1f}° | Offsets: P{self.pitch_offset:+.1f}° R{self.roll_offset:+.1f}°")
            
            time.sleep(0.2)  # 5Hz sampling
        
        if len(pitch_readings) < samples // 2:
            print(f"❌ Too few successful readings ({len(pitch_readings)}/{samples})")
            return False
        
        # Calculate statistics
        pitch_mean = np.mean(pitch_readings)
        roll_mean = np.mean(roll_readings)
        pitch_std = np.std(pitch_readings)
        roll_std = np.std(roll_readings)
        
        print(f"\n📊 Calibration Check Results:")
        print(f"   📐 Calibrated Pitch: {pitch_mean:+.2f}° ± {pitch_std:.2f}° (should be ~0°)")
        print(f"   📐 Calibrated Roll:  {roll_mean:+.2f}° ± {roll_std:.2f}° (should be ~0°)")
        print(f"   🔧 Applied offsets: P{self.pitch_offset:+.2f}° R{self.roll_offset:+.2f}°")
        print()
        
        # Check if within tolerance
        pitch_ok = abs(pitch_mean) < tolerance_degrees
        roll_ok = abs(roll_mean) < tolerance_degrees
        noise_ok = pitch_std < 0.5 and roll_std < 0.5
        
        if pitch_ok and roll_ok and noise_ok:
            print("✅ CALIBRATION IS EXCELLENT!")
            print(f"   • Calibrated angles are within ±{tolerance_degrees}° of level")
            print(f"   • Noise levels are very low (< 0.5°)")
            print(f"   • Your IMU calibration is working perfectly")
            return True
        elif abs(pitch_mean) < 0.1 and abs(roll_mean) < 0.1 and noise_ok:
            print("🎯 CALIBRATION IS PERFECT!")
            print(f"   • Calibrated readings are essentially zero (< 0.1°)")
            print(f"   • This means your table is actually level AND your offsets are correct")
            print(f"   • OR your IMU is reading true zeros and needs no calibration")
            print(f"   • IMU is working excellently for ball balancing!")
            return True
        elif noise_ok and abs(pitch_mean) < 2.0 and abs(roll_mean) < 2.0:
            print("🟡 CALIBRATION IS GOOD:")
            print(f"   • Calibrated angles within acceptable range (< 2°)")
            print(f"   • Low noise indicates stable readings")
            print(f"   • Should work fine for ball balancing")
            print(f"   • Consider recalibration only if you need higher precision")
            return True
        else:
            print("⚠️ CALIBRATION NEEDS ATTENTION:")
            if not pitch_ok:
                print(f"   • Calibrated pitch offset: {pitch_mean:+.2f}° (should be < ±{tolerance_degrees}°)")
            if not roll_ok:
                print(f"   • Calibrated roll offset: {roll_mean:+.2f}° (should be < ±{tolerance_degrees}°)")
            if not noise_ok:
                print(f"   • High noise: P±{pitch_std:.2f}° R±{roll_std:.2f}° (should be < 0.5°)")
            print(f"   • Recommend running full recalibration")
            return False
    
    def quick_drift_test(self, duration=30):
        """Quick test to check for IMU drift over time"""
        print(f"\n🕐 IMU Drift Test ({duration} seconds)")
        print("=" * 50)
        print(f"Keep the table level and stationary for {duration} seconds...")
        
        input("Press Enter to start drift test...")
        
        start_time = time.time()
        readings = []
        
        print("Time | Pitch | Roll  | Note")
        print("-" * 35)
        
        while time.time() - start_time < duration:
            reading = self.get_imu_reading()
            if reading:
                heading, pitch, roll = reading
                
                # Apply calibration
                calibrated_pitch = pitch + self.pitch_offset
                calibrated_roll = roll + self.roll_offset
                
                elapsed = time.time() - start_time
                readings.append((elapsed, calibrated_pitch, calibrated_roll))
                
                # Print every 5 seconds
                if len(readings) % 25 == 0:  # Every 5 seconds at 5Hz
                    print(f"{elapsed:4.0f}s | {calibrated_pitch:+5.1f}° | {calibrated_roll:+5.1f}° |")
        
        if len(readings) < 10:
            print("❌ Not enough readings for drift analysis")
            return
        
        # Analyze drift
        times = [r[0] for r in readings]
        pitches = [r[1] for r in readings]
        rolls = [r[2] for r in readings]
        
        pitch_drift = pitches[-1] - pitches[0]
        roll_drift = rolls[-1] - rolls[0]
        
        print(f"\n📊 Drift Analysis:")
        print(f"   📐 Pitch drift: {pitch_drift:+.2f}° over {duration}s")
        print(f"   📐 Roll drift:  {roll_drift:+.2f}° over {duration}s")
        
        if abs(pitch_drift) < 0.5 and abs(roll_drift) < 0.5:
            print("✅ DRIFT IS ACCEPTABLE (< 0.5° over 30s)")
        else:
            print("⚠️ SIGNIFICANT DRIFT DETECTED")
            print("   • IMU may need recalibration or environment is not stable")
    
    def cleanup(self):
        """Clean up connections"""
        if self.imu_interface:
            self.imu_interface.cleanup()
    
    def run_full_check(self):
        """Run complete calibration check"""
        print("🔍 IMU Calibration Check Tool")
        print("=" * 60)
        print()
        
        # Load existing calibration
        if not self.load_existing_calibration():
            print("\n💡 No valid calibration found. Run:")
            print("   python embedded_imu_calibration.py --port COM7")
            return
        
        # Connect to IMU
        if not self.connect_imu():
            return
        
        try:
            # Check accuracy
            accuracy_ok = self.check_calibration_accuracy()
            
            if accuracy_ok:
                # Run drift test if accuracy is good
                print(f"\n🎯 Since calibration looks good, let's check for drift...")
                self.quick_drift_test()
            
            print(f"\n🏁 Calibration Check Complete!")
            
            if accuracy_ok:
                print("✅ Your calibration is working well!")
                print("💡 You can use your IMU with confidence.")
            else:
                print("⚠️ Calibration needs improvement.")
                print("💡 Run: python embedded_imu_calibration.py --port COM7")
            
        except KeyboardInterrupt:
            print("\n🛑 Check interrupted by user")
        finally:
            self.cleanup()

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Quick IMU Calibration Check")
    parser.add_argument("--port", default="COM7", help="IMU COM port")
    parser.add_argument("--samples", type=int, default=20, help="Number of samples for accuracy check")
    parser.add_argument("--tolerance", type=float, default=1.0, help="Tolerance in degrees for 'good' calibration")
    
    args = parser.parse_args()
    
    checker = CalibrationChecker(args.port)
    checker.run_full_check()

if __name__ == "__main__":
    main()
