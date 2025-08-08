"""
Servo Controller for Ball Balancing Table
Converts table angles to servo positions and controls Dynamixel servos
"""

from dynamixel_sdk import *
import time
import math
import csv
import numpy as np

class ServoController:
    def __init__(self, device_name='COM5', baudrate=1000000, servo_ids=[1, 2]):
        # Setup Parameters
        self.device_name = device_name
        self.baudrate = baudrate
        self.servo_ids = servo_ids

        self.pitch_calibration = None
        self.roll_calibration = None
        self.use_calibration = False
        
        # Servo register addresses (XM430-W350 or similar)
        self.ADDR_TORQUE_ENABLE = 64
        self.ADDR_GOAL_POSITION = 116
        self.ADDR_PRESENT_POSITION = 132
        self.PROTOCOL_VERSION = 2.0
        
        # Servo limits and conversion
        self.DXL_MIN_POS = 1707
        self.DXL_MAX_POS = 2390
        # Per-servo center positions (ID 1: 1990, ID 2: 2010)
        self.DXL_CENTER_POSITIONS = [1990, 2010]  # Index 0: ID 1, Index 1: ID 2
        self.DXL_HALF_RANGE = (self.DXL_MAX_POS - self.DXL_MIN_POS) // 2  # 1536
        
        # Kinematic conversion: half range = 11 degrees = 0.1920 radians
        self.MAX_TABLE_ANGLE_RAD = math.radians(11.0)  # ±11 degrees
        self.STEPS_PER_RADIAN = self.DXL_HALF_RANGE / self.MAX_TABLE_ANGLE_RAD  # ~29,325
        print(self.STEPS_PER_RADIAN)
        
        # SDK objects
        self.port_handler = None
        self.packet_handler = None
        self.group_sync_write = None
        
        # Connection status
        self.connected = False
        
        print(f"🤖 Servo Controller initialized:")
        print(f"   Range: {self.DXL_MIN_POS} to {self.DXL_MAX_POS} (centers: {self.DXL_CENTER_POSITIONS})")
        print(f"   Max table angle: ±{math.degrees(self.MAX_TABLE_ANGLE_RAD):.1f}°")
        print(f"   Conversion: {self.STEPS_PER_RADIAN:.0f} steps/radian")

    def load_calibration_data(self, pitch_file='servo_calib_pitch.csv', roll_file='servo_calib_roll.csv'):
        """Load calibration CSV files and enable calibrated mapping"""
        def load_file(filename):
            with open(filename, 'r') as f:
                reader = csv.DictReader(f)
                positions = []
                angles = []
                for row in reader:
                    positions.append(int(row["ServoPosition"]))
                    angles.append(float(row["Degrees"]))
                return np.array(angles), np.array(positions)  # Interpolation expects x (angle), y (position)

        try:
            self.pitch_calibration = load_file(pitch_file)
            self.roll_calibration = load_file(roll_file)
            self.use_calibration = True
            print(f"📈 Loaded calibration data from '{pitch_file}' and '{roll_file}'")
        except Exception as e:
            print(f"⚠️ Failed to load calibration: {e}")
            self.use_calibration = False
    
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
    
    def angle_to_servo_position(self, angle_rad, servo_index):
        """Convert angle to servo position using calibration or default mapping"""
        angle_deg = math.degrees(angle_rad)
        if self.use_calibration:
            if servo_index == 0 and self.pitch_calibration:
                angles, positions = self.pitch_calibration
            elif servo_index == 1 and self.roll_calibration:
                angles, positions = self.roll_calibration
            else:
                raise ValueError("Calibration data missing for servo index", servo_index)

            # Clamp to calibration range
            angle_deg = max(min(angle_deg, max(angles)), min(angles))
            return int(np.interp(angle_deg, angles, positions))
        else:
            # Default linear mapping
            angle_rad = max(-self.MAX_TABLE_ANGLE_RAD, min(self.MAX_TABLE_ANGLE_RAD, angle_rad))
            steps_from_center = int(angle_rad * self.STEPS_PER_RADIAN)
            center_pos = self.DXL_CENTER_POSITIONS[servo_index]
            servo_position = center_pos + steps_from_center
            return max(self.DXL_MIN_POS, min(self.DXL_MAX_POS, servo_position))
    
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
            # Hardware mapping: servo_ids[0] (ID 1) controls pitch, servo_ids[1] (ID 2) controls roll
            # Note: ID 2 has inverted sign - positive servo command = negative roll
            pitch_pos = self.angle_to_servo_position(pitch_rad, 0)
            roll_pos = self.angle_to_servo_position(-roll_rad, 1)  # Invert roll for hardware
            # Clear previous parameters
            self.group_sync_write.clearParam()
            # Add servo commands - correct order: [pitch_pos, roll_pos] for [ID1, ID2]
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
            (5, 0, "Pitch forward 5°"),
            (-5, 0, "Pitch backward 5°"),
            (0, 5, "Roll right 5°"),
            (0, -5, "Roll left 5°"),
            (5, 5, "Diagonal 5°"),
            (0, 0, "Return to center")
        ]
        
        for pitch, roll, description in movements:
            print(f"Moving to: {description}")
            controller.set_table_angles_degrees(pitch, roll)
            input("Press Enter to continue...")
        
        print("✅ Test completed successfully")
        
    except KeyboardInterrupt:
        print("🛑 Test interrupted")
    
    finally:
        controller.disconnect()

def test_servo_circular_motion():
    """Test the servo controller with a circular motion of 10° radius in the pitch/roll plane."""
    import math
    controller = ServoController()
    if not controller.connect():
        return
    try:
        print("🎯 Testing circular motion (10° radius)...")
        steps = 30  # Number of points around the circle
        radius = 11  # degrees
        for i in range(steps):
            angle = 2 * math.pi * i / steps
            pitch = radius * math.cos(angle)
            roll = radius * math.sin(angle)
            print(f"Setting pitch {pitch:.2f}°, roll {roll:.2f}°")
            controller.set_table_angles_degrees(pitch, roll)
            time.sleep(0.0166)
        print("✅ Circular motion test completed successfully")
    except KeyboardInterrupt:
        print("🛑 Test interrupted")
    finally:
        controller.set_table_angles_degrees(0, 0)  # Return to center
        time.sleep(1)
        print("Returning to center position")
        controller.disconnect()


if __name__ == "__main__":
    
    test_servo_controller()
    #test_servo_circular_motion()
