#!/usr/bin/env python3
"""
Embedded IMU Calibration for Ball Balancing Table

This script provides specialized calibration for BNO055 IMUs that are embedded
inside ball balancing tables, where traditional calibration methods aren't possible.
"""

import sys
import time
import numpy as np
from datetime import datetime

# Add IMU interface
sys.path.append('imu')
try:
    from imu_simple import SimpleBNO055Interface
    IMU_AVAILABLE = True
except ImportError:
    IMU_AVAILABLE = False
    print("❌ IMU interface not available")

class EmbeddedIMUCalibrator:
    """Specialized calibration for embedded table IMUs"""
    
    def __init__(self, imu_port="COM7"):
        self.imu_port = imu_port
        self.imu = None
        self.connected = False
        
        # Calibration data storage
        self.gyro_offset_x = 0.0
        self.gyro_offset_y = 0.0  
        self.gyro_offset_z = 0.0
        self.level_pitch_offset = 0.0
        self.level_roll_offset = 0.0
        
        # Known angle calibration points
        self.calibration_points = []
        
    def connect(self):
        """Connect to embedded IMU"""
        if not IMU_AVAILABLE:
            print("❌ IMU interface not available")
            return False
            
        print(f"🧭 Connecting to embedded IMU on {self.imu_port}...")
        self.imu = SimpleBNO055Interface(port=self.imu_port)
        
        if self.imu.connect():
            self.connected = True
            print("✅ Connected to embedded IMU")
            time.sleep(2)  # Let IMU stabilize
            return True
        else:
            print("❌ Failed to connect to embedded IMU")
            return False
    
    def calibrate_gyro_bias(self, duration=30):
        """Calibrate gyroscope bias - table must be perfectly still"""
        if not self.connected:
            print("❌ IMU not connected")
            return False
        
        print("🎯 Gyroscope Bias Calibration")
        print("=" * 40)
        print("📋 IMPORTANT: Keep your table PERFECTLY STILL for 30 seconds")
        print("   • No touching the table")
        print("   • No vibrations or movement")
        print("   • This measures gyroscope drift/bias")
        print()
        
        input("Press Enter when table is completely still...")
        
        print(f"📊 Collecting gyroscope bias data for {duration} seconds...")
        
        gyro_x_samples = []
        gyro_y_samples = []
        gyro_z_samples = []
        
        start_time = time.time()
        sample_count = 0
        
        while time.time() - start_time < duration:
            line = self.imu.read_line()
            if line and line.startswith("GYRO:"):
                # If your Arduino sends gyro data separately
                try:
                    data_part = line[5:]
                    gx, gy, gz = map(float, data_part.split(','))
                    gyro_x_samples.append(gx)
                    gyro_y_samples.append(gy) 
                    gyro_z_samples.append(gz)
                    sample_count += 1
                    
                    if sample_count % 100 == 0:
                        print(f"   {sample_count} samples collected...")
                        
                except:
                    pass
            
            time.sleep(0.01)  # 100Hz sampling
        
        if len(gyro_x_samples) < 100:
            print("❌ Insufficient gyro data - check Arduino code sends GYRO: data")
            return False
        
        # Calculate offsets
        self.gyro_offset_x = np.mean(gyro_x_samples)
        self.gyro_offset_y = np.mean(gyro_y_samples)
        self.gyro_offset_z = np.mean(gyro_z_samples)
        
        # Calculate noise levels
        gyro_noise_x = np.std(gyro_x_samples)
        gyro_noise_y = np.std(gyro_y_samples)
        gyro_noise_z = np.std(gyro_z_samples)
        
        print("✅ Gyroscope bias calibration complete!")
        print(f"   📐 X-axis bias: {self.gyro_offset_x:+.3f}°/s (noise: {gyro_noise_x:.3f})")
        print(f"   📐 Y-axis bias: {self.gyro_offset_y:+.3f}°/s (noise: {gyro_noise_y:.3f})")
        print(f"   📐 Z-axis bias: {self.gyro_offset_z:+.3f}°/s (noise: {gyro_noise_z:.3f})")
        
        if max(gyro_noise_x, gyro_noise_y, gyro_noise_z) > 0.1:
            print("⚠️ High gyroscope noise detected - table may not be stable enough")
        
        return True
    
    def calibrate_level_reference(self):
        """Calibrate level reference - table at exactly 0° pitch and roll"""
        if not self.connected:
            print("❌ IMU not connected")
            return False
        
        print("📐 Level Reference Calibration")
        print("=" * 40)
        print("📋 CRITICAL: Adjust your table to be PERFECTLY LEVEL")
        print("   • Use a bubble level or laser level")
        print("   • Both pitch and roll must be exactly 0°")
        print("   • This becomes your zero reference")
        print()
        
        input("Press Enter when table is perfectly level...")
        
        print("📊 Measuring level reference for 10 seconds...")
        
        pitch_samples = []
        roll_samples = []
        sample_count = 0
        
        start_time = time.time()
        while time.time() - start_time < 10:
            line = self.imu.read_line()
            if line and line.startswith("DATA:"):
                try:
                    data_part = line[6:]
                    heading, pitch, roll = map(float, data_part.split(','))
                    pitch_samples.append(pitch)
                    roll_samples.append(roll)
                    sample_count += 1
                    
                    if sample_count % 50 == 0:
                        print(f"   Sample {sample_count}: P{pitch:+.1f}° R{roll:+.1f}°")
                        
                except:
                    pass
            time.sleep(0.01)
        
        if len(pitch_samples) < 50:
            print("❌ Insufficient samples")
            return False
        
        # Calculate level offsets
        self.level_pitch_offset = np.mean(pitch_samples)
        self.level_roll_offset = np.mean(roll_samples)
        
        pitch_std = np.std(pitch_samples)
        roll_std = np.std(roll_samples)
        
        print("✅ Level reference calibration complete!")
        print(f"   📐 Pitch offset: {self.level_pitch_offset:+.2f}° (std: {pitch_std:.2f}°)")
        print(f"   📐 Roll offset:  {self.level_roll_offset:+.2f}° (std: {roll_std:.2f}°)")
        
        if max(pitch_std, roll_std) > 0.3:
            print("⚠️ High variation - table may not be stable or level")
        
        return True
    
    def calibrate_known_angles(self):
        """Calibrate at known tilt angles for linearity verification"""
        if not self.connected:
            print("❌ IMU not connected")
            return False
        
        print("📊 Known Angle Calibration")
        print("=" * 40)
        print("📋 This will verify IMU linearity at known angles")
        print("   • Use a digital angle finder or protractor")
        print("   • We'll test several known angles")
        print()
        
        test_angles = [0, 5, 10, -5, -10]  # Degrees
        
        for target_angle in test_angles:
            print(f"🎯 Set table to exactly {target_angle:+}° pitch (roll = 0°)")
            print("   Use a digital angle finder for accuracy")
            input("   Press Enter when set...")
            
            # Collect samples at this angle
            print(f"📊 Measuring for 5 seconds at {target_angle:+}°...")
            pitch_samples = []
            
            start_time = time.time()
            while time.time() - start_time < 5:
                line = self.imu.read_line()
                if line and line.startswith("DATA:"):
                    try:
                        data_part = line[6:]
                        heading, pitch, roll = map(float, data_part.split(','))
                        # Apply level offset
                        corrected_pitch = pitch - self.level_pitch_offset
                        pitch_samples.append(corrected_pitch)
                    except:
                        pass
                time.sleep(0.01)
            
            if pitch_samples:
                measured_angle = np.mean(pitch_samples)
                angle_std = np.std(pitch_samples)
                error = measured_angle - target_angle
                
                self.calibration_points.append({
                    'target': target_angle,
                    'measured': measured_angle,
                    'error': error,
                    'std': angle_std
                })
                
                print(f"   📐 Target: {target_angle:+.0f}°, Measured: {measured_angle:+.1f}°, Error: {error:+.1f}° (std: {angle_std:.2f}°)")
        
        # Analyze linearity
        if len(self.calibration_points) >= 3:
            targets = [p['target'] for p in self.calibration_points]
            measured = [p['measured'] for p in self.calibration_points]
            errors = [p['error'] for p in self.calibration_points]
            
            # Linear fit
            coeffs = np.polyfit(targets, measured, 1)
            slope, intercept = coeffs
            
            print("\n📊 Linearity Analysis:")
            print(f"   📐 Slope: {slope:.3f} (ideal: 1.000)")
            print(f"   📐 Intercept: {intercept:+.2f}° (ideal: 0.00°)")
            print(f"   📐 Max error: {max(errors, key=abs):+.1f}°")
            print(f"   📐 RMS error: {np.sqrt(np.mean(np.array(errors)**2)):.2f}°")
            
            if abs(slope - 1.0) > 0.05:
                print("⚠️ Significant linearity error detected")
            if abs(intercept) > 1.0:
                print("⚠️ Significant offset error detected")
        
        return True
    
    def save_calibration(self, filename="embedded_imu_calibration.txt"):
        """Save calibration data to file"""
        try:
            with open(filename, 'w') as f:
                f.write("# Embedded IMU Calibration Data\n")
                f.write(f"# Generated: {datetime.now()}\n")
                f.write(f"# IMU Port: {self.imu_port}\n\n")
                
                f.write("# Level Reference Offsets\n")
                f.write(f"level_pitch_offset = {self.level_pitch_offset:.4f}\n")
                f.write(f"level_roll_offset = {self.level_roll_offset:.4f}\n\n")
                
                f.write("# Gyroscope Bias Offsets\n")
                f.write(f"gyro_offset_x = {self.gyro_offset_x:.4f}\n")
                f.write(f"gyro_offset_y = {self.gyro_offset_y:.4f}\n")
                f.write(f"gyro_offset_z = {self.gyro_offset_z:.4f}\n\n")
                
                if self.calibration_points:
                    f.write("# Known Angle Calibration Points\n")
                    for i, point in enumerate(self.calibration_points):
                        f.write(f"# Point {i+1}: Target {point['target']:+.0f}°, "
                               f"Measured {point['measured']:+.1f}°, "
                               f"Error {point['error']:+.1f}°\n")
            
            print(f"✅ Calibration saved to {filename}")
            return True
        except Exception as e:
            print(f"❌ Failed to save calibration: {e}")
            return False
    
    def run_full_calibration(self):
        """Run complete embedded IMU calibration sequence"""
        print("🚀 Embedded IMU Calibration for Ball Balancing Table")
        print("=" * 60)
        print()
        print("📋 This calibration is designed for IMUs embedded inside tables")
        print("   where traditional motion-based calibration isn't possible.")
        print()
        
        if not self.connect():
            return False
        
        try:
            # Step 1: Level reference
            if not self.calibrate_level_reference():
                print("❌ Level reference calibration failed")
                return False
            
            print("\n" + "="*50)
            
            # Step 2: Optional known angles (for advanced users)
            print("\n🎯 Advanced Calibration (Optional)")
            do_advanced = input("Do known angle calibration? (y/n): ").lower() == 'y'
            
            if do_advanced:
                if not self.calibrate_known_angles():
                    print("❌ Known angle calibration failed")
                    return False
            
            # Step 3: Save results
            if not self.save_calibration():
                print("❌ Failed to save calibration")
                return False
            
            print("\n🎉 Embedded IMU Calibration Complete!")
            print("📊 Results Summary:")
            print(f"   📐 Level pitch offset: {self.level_pitch_offset:+.2f}°")
            print(f"   📐 Level roll offset:  {self.level_roll_offset:+.2f}°")
            
            print("\n💡 Next Steps:")
            print("   1. Use these offsets in your ball balancing control")
            print("   2. Test with: python compare_control.py --imu-control")
            print("   3. Your BNO055 should now perform much better!")
            
            return True
            
        except KeyboardInterrupt:
            print("\n🛑 Calibration cancelled by user")
            return False
        finally:
            if self.imu:
                self.imu.cleanup()

def main():
    """Main calibration function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Embedded IMU Calibration")
    parser.add_argument("--port", default="COM7", help="IMU COM port")
    args = parser.parse_args()
    
    calibrator = EmbeddedIMUCalibrator(args.port)
    calibrator.run_full_calibration()

if __name__ == "__main__":
    main()
