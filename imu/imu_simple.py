"""
Simple BNO055 IMU Interface for Ball Balancing

A clean, simple interface that just displays IMU data without calibration complexity.
Perfect for ball balancing applications where you just need consistent angle readings.

Usage: python imu_simple.py [COM_PORT]
Press Ctrl+C to exit
"""

import serial
import time
import sys
from datetime import datetime

class SimpleBNO055Interface:
    """Simple BNO055 interface focused on ball balancing needs"""
    
    def __init__(self, port: str = 'COM6', baud_rate: int = 115200):
        self.port = port
        self.baud_rate = baud_rate
        self.ser = None
        self.connected = False
        self.data_count = 0
        
    def connect(self) -> bool:
        """Connect to Arduino"""
        try:
            print(f"🔌 Connecting to {self.port} at {self.baud_rate} baud...")
            self.ser = serial.Serial(self.port, self.baud_rate, timeout=0.1)
            time.sleep(3)  # Wait for Arduino reset
            
            print("📡 Waiting for Arduino initialization...")
            self.connected = True
            return True
            
        except Exception as e:
            print(f"❌ Connection failed: {e}")
            return False
    
    def read_line(self) -> str:
        """Read line from serial"""
        try:
            if self.ser and self.ser.is_open and self.ser.in_waiting > 0:
                line = self.ser.readline().decode('utf-8', errors='ignore').strip()
                return line
        except:
            pass
        return ""
    
    def run_interface(self):
        """Main interface loop"""
        print("\n🧭 Simple BNO055 Interface for Ball Balancing")
        print("=" * 60)
        print("📋 This interface displays:")
        print("  • Real-time IMU data (heading, pitch, roll)")
        print("  • 100Hz data rate for smooth ball balancing control")
        print("  • No calibration complexity - just works!")
        print("📝 Press Ctrl+C to exit")
        print("=" * 60)
        
        if not self.connect():
            return
        
        print("\n🚀 Monitoring started...")
        print("💡 Focus on PITCH (forward/back) and ROLL (left/right) for ball balancing")
        print("-" * 80)
        
        try:
            last_display_time = 0
            
            while True:
                line = self.read_line()
                if line:
                    current_time = time.time()
                    
                    if line.startswith("DATA: "):
                        # Parse IMU data
                        try:
                            data_part = line[6:]
                            heading, pitch, roll = map(float, data_part.split(','))
                            self.data_count += 1
                            
                            # Show every 10th reading to avoid overwhelming display
                            if self.data_count % 10 == 0:
                                timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
                                
                                # Highlight pitch and roll for ball balancing
                                print(f"{timestamp} | H:{heading:6.1f}° | "
                                      f"🎯 PITCH:{pitch:+6.1f}° ROLL:{roll:+6.1f}° | "
                                      f"#{self.data_count:5d}")
                                
                        except:
                            print(f"⚠️ Malformed data: {line}")
                    
                    elif any(prefix in line for prefix in ["INIT:", "READY:", "INFO:"]):
                        timestamp = datetime.now().strftime("%H:%M:%S")
                        print(f"{timestamp} | 📋 {line}")
                    
                    # Show helpful tips periodically
                    if current_time - last_display_time > 30:  # Every 30 seconds
                        last_display_time = current_time
                        print("\n💡 Ball Balancing Tips:")
                        print("   • PITCH controls forward/backward ball movement")
                        print("   • ROLL controls left/right ball movement")
                        print("   • Use these angles for servo control feedback")
                        print("-" * 80)
                
                time.sleep(0.001)  # Very small delay for CPU efficiency
        
        except KeyboardInterrupt:
            print(f"\n🛑 Monitoring stopped by user")
            print(f"📊 Total data points received: {self.data_count}")
            print(f"📈 Effective data rate: ~{self.data_count/30:.0f}Hz (displaying every 10th)")
        
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources"""
        if self.ser and self.ser.is_open:
            try:
                self.ser.close()
            except:
                pass
        print("🔌 Connection closed")

def main():
    """Main function"""
    port = sys.argv[1] if len(sys.argv) > 1 else 'COM6'
    
    interface = SimpleBNO055Interface(port)
    interface.run_interface()

if __name__ == "__main__":
    main()
