"""
Servo Controller for Ball Balancing Table
Converts table angles to servo positions and controls Dynamixel servos
"""

from dynamixel_sdk import *
import time
import math

class ServoController:
    def __init__(self, device_name='COM5', baudrate=1000000, servo_ids=[1, 2]):
        # Setup Parameters
        self.device_name = device_name
        self.baudrate = baudrate
        self.servo_ids = servo_ids
        
        # Servo register addresses (XM430-W350 or similar)
        self.ADDR_TORQUE_ENABLE = 64
        self.ADDR_GOAL_POSITION = 116
        self.ADDR_PRESENT_POSITION = 132
        self.PROTOCOL_VERSION = 2.0
        
        # Servo limits and conversion
        self.DXL_MIN_POS = 1023
        self.DXL_MAX_POS = 3073
        self.DXL_CENTER_POS = (self.DXL_MIN_POS + self.DXL_MAX_POS) // 2  # 2048
        self.DXL_HALF_RANGE = (self.DXL_MAX_POS - self.DXL_MIN_POS) // 2  # 1536
        
        # Kinematic conversion: half range = 3 degrees = 0.0524 radians
        self.MAX_TABLE_ANGLE_RAD = math.radians(3.0)  # ±3 degrees
        self.STEPS_PER_RADIAN = self.DXL_HALF_RANGE / self.MAX_TABLE_ANGLE_RAD  # ~29,325
        
        # SDK objects
        self.port_handler = None
        self.packet_handler = None
        self.group_sync_write = None
        
        # Connection status
        self.connected = False
        
        print(f"🤖 Servo Controller initialized:")
        print(f"   Range: {self.DXL_MIN_POS} to {self.DXL_MAX_POS} (center: {self.DXL_CENTER_POS})")
        print(f"   Max table angle: ±{math.degrees(self.MAX_TABLE_ANGLE_RAD):.1f}°")
        print(f"   Conversion: {self.STEPS_PER_RADIAN:.0f} steps/radian")
    
    def connect(self):
        """Initialize and connect to servos"""
        try:
            # Initialize SDK
            self.port_handler = PortHandler(self.device_name)
            self.packet_handler = PacketHandler(self.PROTOCOL_VERSION)
            self.group_sync_write = GroupSyncWrite(
                self.port_handler, self.packet_handler, self.ADDR_GOAL_POSITION, 4
            )
            
            # Open port
            if not self.port_handler.openPort():
                print(f"❌ Failed to open port {self.device_name}")
                return False
            
            # Set baudrate
            if not self.port_handler.setBaudRate(self.baudrate):
                print(f"❌ Failed to set baud rate to {self.baudrate}")
                return False
            
            # Enable torque on all servos
            for servo_id in self.servo_ids:
                result, error = self.packet_handler.write1ByteTxRx(
                    self.port_handler, servo_id, self.ADDR_TORQUE_ENABLE, 1
                )
                if result != COMM_SUCCESS:
                    print(f"❌ Failed to enable torque on servo {servo_id}")
                    return False
            
            self.connected = True
            print(f"✅ Connected to servos {self.servo_ids}")
            
            # Move to center position
            self.set_table_angles(0.0, 0.0)
            time.sleep(1)  # Allow servos to move to center
            
            return True
            
        except Exception as e:
            print(f"❌ Connection error: {e}")
            return False
    
    def disconnect(self):
        """Disable servos and close connection"""
        if self.connected and self.port_handler:
            # Disable torque on all servos
            for servo_id in self.servo_ids:
                self.packet_handler.write1ByteTxRx(
                    self.port_handler, servo_id, self.ADDR_TORQUE_ENABLE, 0
                )
            
            # Close port
            self.port_handler.closePort()
            self.connected = False
            print("🔌 Servos disconnected")
    
    def angle_to_servo_position(self, angle_rad):
        """Convert table angle (radians) to servo position"""
        # Clamp angle to safe range
        angle_rad = max(-self.MAX_TABLE_ANGLE_RAD, min(self.MAX_TABLE_ANGLE_RAD, angle_rad))
        
        # Convert to servo steps
        steps_from_center = int(angle_rad * self.STEPS_PER_RADIAN)
        servo_position = self.DXL_CENTER_POS + steps_from_center
        
        # Clamp to servo limits
        servo_position = max(self.DXL_MIN_POS, min(self.DXL_MAX_POS, servo_position))
        
        return servo_position
    
    def to_little_endian_bytes(self, value):
        """Convert integer to 4-byte little-endian array"""
        return [
            value & 0xFF,
            (value >> 8) & 0xFF,
            (value >> 16) & 0xFF,
            (value >> 24) & 0xFF
        ]
    
    def set_table_angles(self, pitch_rad, roll_rad):
        """
        Set table angles in radians
        pitch_rad: forward/backward tilt (positive = forward down)
        roll_rad: left/right tilt (positive = right down)
        """
        if not self.connected:
            print("⚠️ Servos not connected")
            return False
        
        try:
            # Convert angles to servo positions
            # Assuming servo_ids[0] controls pitch, servo_ids[1] controls roll
            pitch_pos = self.angle_to_servo_position(pitch_rad)
            roll_pos = self.angle_to_servo_position(roll_rad)
            
            # Clear previous parameters
            self.group_sync_write.clearParam()
            
            # Add servo commands
            servo_positions = [pitch_pos, roll_pos]
            for i, servo_id in enumerate(self.servo_ids):
                param = self.to_little_endian_bytes(servo_positions[i])
                if not self.group_sync_write.addParam(servo_id, param):
                    print(f"❌ Failed to add param for servo {servo_id}")
                    return False
            
            # Send commands simultaneously
            result = self.group_sync_write.txPacket()
            if result != COMM_SUCCESS:
                print(f"❌ Failed to send servo commands: {self.packet_handler.getTxRxResult(result)}")
                return False
            
            return True
            
        except Exception as e:
            print(f"❌ Servo control error: {e}")
            return False
    
    def set_table_angles_degrees(self, pitch_deg, roll_deg):
        """Convenience method to set angles in degrees"""
        return self.set_table_angles(math.radians(pitch_deg), math.radians(roll_deg))
    
    def get_max_angle_degrees(self):
        """Get maximum table angle in degrees"""
        return math.degrees(self.MAX_TABLE_ANGLE_RAD)


# Test function
def test_servo_controller():
    """Test the servo controller with some basic movements"""
    controller = ServoController()
    
    if not controller.connect():
        return
    
    try:
        print("🎯 Testing servo movements...")
        
        # Test sequence
        movements = [
            (0, 0, "Center"),
            (2, 0, "Pitch forward 2°"),
            (-2, 0, "Pitch backward 2°"),
            (0, 2, "Roll right 2°"),
            (0, -2, "Roll left 2°"),
            (1, 1, "Diagonal 1°"),
            (0, 0, "Return to center")
        ]
        
        for pitch, roll, description in movements:
            print(f"Moving to: {description}")
            controller.set_table_angles_degrees(pitch, roll)
            time.sleep(2)
        
        print("✅ Test completed successfully")
        
    except KeyboardInterrupt:
        print("🛑 Test interrupted")
    
    finally:
        controller.disconnect()


if __name__ == "__main__":
    test_servo_controller()
