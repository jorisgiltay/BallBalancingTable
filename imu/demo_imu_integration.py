#!/usr/bin/env python3
"""
Demo script showing IMU integration with ball balancing control

This script demonstrates the IMU feedback integration without requiring
actual hardware connection.
"""

import sys
import time
import threading
from datetime import datetime

# Mock IMU data for demonstration
class MockIMUInterface:
    """Mock IMU interface that simulates IMU data for demonstration"""
    
    def __init__(self, port="COM6"):
        self.port = port
        self.connected = False
        self.pitch = 0.0
        self.roll = 0.0
        self.heading = 0.0
        
    def connect(self):
        print(f"🧭 Mock IMU connecting to {self.port}...")
        time.sleep(1)
        self.connected = True
        print("✅ Mock IMU connected (simulated)")
        return True
    
    def read_line(self):
        """Simulate IMU data that matches table movement"""
        if self.connected:
            # Simulate some gentle oscillation
            t = time.time()
            self.pitch = 2.0 * (1 + 0.5 * (t % 10 - 5))  # Gentle pitch movement
            self.roll = 1.5 * (1 + 0.3 * ((t * 1.3) % 8 - 4))  # Gentle roll movement
            self.heading = 45.0  # Fixed heading
            
            return f"DATA: {self.heading:.1f},{self.pitch:.1f},{self.roll:.1f}"
        return ""
    
    def cleanup(self):
        print("🧭 Mock IMU disconnected")
        self.connected = False

def demo_imu_integration():
    """Demonstrate IMU integration features"""
    print("🚀 Ball Balancing Control - IMU Integration Demo")
    print("=" * 60)
    print()
    print("📋 This demo shows how IMU feedback is integrated into the ball balancing system:")
    print("   • Real-time IMU angle monitoring")
    print("   • Comparison between simulation and real hardware angles")
    print("   • Background thread for continuous IMU data reading")
    print("   • Thread-safe data sharing with main control loop")
    print()
    
    # Create mock IMU interface
    print("🧭 Setting up mock IMU interface...")
    mock_imu = MockIMUInterface("COM6")
    
    if not mock_imu.connect():
        print("❌ Failed to connect to mock IMU")
        return
    
    print()
    print("📊 Mock IMU Data Stream (simulating real BNO055 output):")
    print("-" * 60)
    
    # Simulate IMU data reading for a few seconds
    start_time = time.time()
    data_count = 0
    
    try:
        while time.time() - start_time < 10:  # Run for 10 seconds
            line = mock_imu.read_line()
            if line:
                data_count += 1
                
                # Show every 10th reading to avoid overwhelming display
                if data_count % 10 == 0:
                    timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
                    
                    # Parse and display data like the real system
                    try:
                        data_part = line[6:]
                        heading, pitch, roll = map(float, data_part.split(','))
                        
                        print(f"{timestamp} | H:{heading:6.1f}° | "
                              f"🎯 PITCH:{pitch:+6.1f}° ROLL:{roll:+6.1f}° | "
                              f"#{data_count:5d}")
                        
                        # Simulate comparison with table angles
                        if data_count % 50 == 0:  # Every 50 readings
                            sim_pitch = 1.8  # Simulated table pitch
                            sim_roll = 1.2   # Simulated table roll
                            
                            pitch_diff = pitch - sim_pitch
                            roll_diff = roll - sim_roll
                            
                            print(f"         | 📐 Table: P{sim_pitch:+.1f}° R{sim_roll:+.1f}° | "
                                  f"🔄 Diff: P{pitch_diff:+.1f}° R{roll_diff:+.1f}°")
                            
                    except:
                        print(f"⚠️ Malformed data: {line}")
            
            time.sleep(0.01)  # 100Hz simulation
            
    except KeyboardInterrupt:
        print("\\n🛑 Demo stopped by user")
    
    mock_imu.cleanup()
    
    print()
    print("📊 Demo Statistics:")
    print(f"   • Total data points: {data_count}")
    print(f"   • Effective data rate: ~{data_count/10:.0f}Hz")
    print(f"   • Duration: 10 seconds")
    print()
    print("✅ IMU Integration Demo Complete!")
    print()
    print("🔧 Next Steps:")
    print("   1. Connect your BNO055 IMU to Arduino")
    print("   2. Upload the arduino_bno055_simple.ino sketch")
    print("   3. Run: python compare_control.py --imu --visuals")
    print("   4. Watch real-time IMU vs simulation angle comparison!")
    print()
    print("💡 Advanced Usage:")
    print("   • Full hardware mode: --imu --servos --camera real")
    print("   • Hybrid mode: --imu --camera hybrid --visuals")
    print("   • IMU control mode: --imu-control --imu-port COM7")
    print("   • Different COM port: --imu --imu-port COM3")
    print()
    print("🎮 NEW: IMU Control Mode!")
    print("   • Table follows your IMU movements directly")
    print("   • No calibration required - works immediately")
    print("   • Perfect for testing IMU responsiveness")
    print("   • Command: python compare_control.py --imu-control")

if __name__ == "__main__":
    demo_imu_integration()
