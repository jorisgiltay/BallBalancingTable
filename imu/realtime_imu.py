#!/usr/bin/env python3
"""
Real-time IMU Viewer

Simple, smooth real-time display of IMU pitch and roll values.
Works well in Windows PowerShell and other terminals.
"""

import sys
import os
import time
import serial
import threading

class RealtimeIMUViewer:
    def __init__(self, port="COM8", baudrate=115200):
        self.port = port
        self.baudrate = baudrate
        self.ser = None
        self.running = False
        
        # Current values
        self.heading = 0.0
        self.pitch = 0.0
        self.roll = 0.0
        self.calibrated_pitch = 0.0
        self.calibrated_roll = 0.0
        
        # Calibration offsets
        self.pitch_offset = 0.0
        self.roll_offset = 0.0
        
        # Statistics
        self.sample_count = 0
        self.start_time = time.time()
        self.last_update = 0
        
    def load_calibration(self):
        """Load calibration offsets"""
        try:
            possible_files = [
                "embedded_imu_calibration.txt",
                "../embedded_imu_calibration.txt", 
                "imu/embedded_imu_calibration.txt"
            ]
            
            filename = None
            for f in possible_files:
                if os.path.exists(f):
                    filename = f
                    break
            
            if not filename:
                print("⚠️ No calibration file found - using raw values")
                return
            
            with open(filename, 'r') as f:
                lines = f.readlines()
            
            for line in lines:
                line = line.strip()
                if line.startswith('level_pitch_offset'):
                    self.pitch_offset = float(line.split('=')[1].strip())
                elif line.startswith('level_roll_offset'):
                    self.roll_offset = float(line.split('=')[1].strip())
            
            print(f"✅ Loaded calibration: P{self.pitch_offset:+.1f}° R{self.roll_offset:+.1f}°")
            
        except Exception as e:
            print(f"⚠️ Failed to load calibration: {e}")
    
    def connect(self):
        """Connect to IMU"""
        try:
            print(f"🔌 Connecting to IMU on {self.port}...")
            self.ser = serial.Serial(self.port, self.baudrate, timeout=1)
            time.sleep(3)  # Wait for Arduino startup
            self.ser.reset_input_buffer()
            print("✅ Connected successfully!")
            return True
        except Exception as e:
            print(f"❌ Connection failed: {e}")
            return False
    
    def create_tilt_indicator(self, angle, width=20):
        """Create a simple tilt indicator"""
        max_angle = 15
        clamped = max(-max_angle, min(max_angle, angle))
        
        if abs(clamped) < 0.5:
            return "LEVEL".center(width)
        
        # Calculate position
        center = width // 2
        offset = int((clamped / max_angle) * (center - 1))
        pos = center + offset
        
        indicator = ['.'] * width
        indicator[center] = '|'  # Center reference
        
        if pos >= center:
            for i in range(center, min(pos + 1, width)):
                indicator[i] = '>'
        else:
            for i in range(max(pos, 0), center):
                indicator[i] = '<'
        
        return ''.join(indicator)
    
    def read_imu_data(self):
        """Background thread to read IMU data"""
        while self.running:
            try:
                if self.ser and self.ser.in_waiting > 0:
                    line = self.ser.readline().decode('utf-8').strip()
                    
                    if line.startswith("DATA:"):
                        try:
                            data_part = line[5:].strip()
                            h, p, r = map(float, data_part.split(','))
                            
                            # Skip startup zeros
                            if not (h == 0.0 and p == 0.0 and r == 0.0):
                                self.heading = h
                                self.pitch = p
                                self.roll = r
                                
                                # Apply calibration
                                self.calibrated_pitch = p + self.pitch_offset
                                self.calibrated_roll = r + self.roll_offset
                                
                                self.sample_count += 1
                                
                        except ValueError:
                            continue
                            
                time.sleep(0.02)  # 50Hz reading
                
            except Exception as e:
                print(f"\n❌ Read error: {e}")
                break
    
    def run(self):
        """Run the real-time viewer"""
        # Load calibration
        self.load_calibration()
        
        # Connect to IMU
        if not self.connect():
            return
        
        print("\n🎯 Real-time IMU Viewer")
        print("=" * 60)
        print("Tilt your table and watch the values change!")
        print("Press Ctrl+C to exit\n")
        
        # Start background reading
        self.running = True
        read_thread = threading.Thread(target=self.read_imu_data, daemon=True)
        read_thread.start()
        
        try:
            last_sample_count = 0
            
            while True:
                # Only update display when we have new data
                if self.sample_count > last_sample_count:
                    runtime = time.time() - self.start_time
                    
                    # Clear the line and show current values
                    print(f"\r📊 Sample #{self.sample_count:5d} | "
                          f"Raw: P{self.pitch:+6.1f}° R{self.roll:+6.1f}° | "
                          f"Level: P{self.calibrated_pitch:+6.1f}° R{self.calibrated_roll:+6.1f}° | "
                          f"⏱️{runtime:5.1f}s", end="", flush=True)
                    
                    last_sample_count = self.sample_count
                    
                    # Show detailed view every 50 samples
                    if self.sample_count % 50 == 0:
                        print(f"\n")
                        print(f"� Pitch: {self.calibrated_pitch:+6.1f}° [{self.create_tilt_indicator(self.calibrated_pitch)}]")
                        print(f"📐 Roll:  {self.calibrated_roll:+6.1f}° [{self.create_tilt_indicator(self.calibrated_roll)}]")
                        
                        if abs(self.calibrated_pitch) < 1 and abs(self.calibrated_roll) < 1:
                            print("✅ Table is level!")
                        else:
                            print("📐 Table is tilted")
                        print()
                
                time.sleep(0.1)  # 10Hz display update
                
        except KeyboardInterrupt:
            print("\n\n🛑 Stopping IMU viewer...")
            
        finally:
            self.running = False
            if self.ser:
                self.ser.close()
            print("🔌 Disconnected")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Real-time IMU Viewer")
    parser.add_argument("--port", default="COM8", help="IMU COM port")
    parser.add_argument("--baudrate", type=int, default=115200, help="Serial baudrate")
    
    args = parser.parse_args()
    
    viewer = RealtimeIMUViewer(args.port, args.baudrate)
    viewer.run()

if __name__ == "__main__":
    main()
