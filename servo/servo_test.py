from dynamixel_sdk import *
import time

# ----------- Setup Parameters -----------
DEVICENAME = 'COM5'           # Your U2D2 COM port
BAUDRATE = 1000000
DXL_IDs = [1, 2]              # Servo IDs

ADDR_TORQUE_ENABLE = 64
ADDR_GOAL_POSITION = 116
ADDR_PRESENT_POSITION = 132

PROTOCOL_VERSION = 2.0

TORQUE_ENABLE = 1
TORQUE_DISABLE = 0
DXL_MIN_POS = 512             # Adjust based on your servo's range
DXL_MAX_POS = 3584
DXL_MOVING_STATUS_THRESHOLD = 20

# ----------- Initialize SDK -----------
portHandler = PortHandler(DEVICENAME)
packetHandler = PacketHandler(PROTOCOL_VERSION)
groupSyncWrite = GroupSyncWrite(portHandler, packetHandler, ADDR_GOAL_POSITION, 4)  # 4 bytes for position
groupSyncRead = GroupSyncRead(portHandler, packetHandler, ADDR_PRESENT_POSITION, 4)  # 4 bytes for position read

if not portHandler.openPort():
    print("❌ Failed to open port")
    exit()

if not portHandler.setBaudRate(BAUDRATE):
    print("❌ Failed to set baud rate")
    exit()

# Enable torque on both servos
for dxl_id in DXL_IDs:
    packetHandler.write1ByteTxRx(portHandler, dxl_id, ADDR_TORQUE_ENABLE, TORQUE_ENABLE)
    groupSyncRead.addParam(dxl_id)  # Add servo to GroupSyncRead

# ----------- Helper to convert int to byte list -----------
def to_little_endian_bytes(value):
    return [value & 0xFF,
            (value >> 8) & 0xFF,
            (value >> 16) & 0xFF,
            (value >> 24) & 0xFF]

# ----------- Movement Loop -----------
positions = [DXL_MIN_POS, DXL_MAX_POS]
index = 0

try:
    while True:
        goal_pos = positions[index]
        index = 1 - index  # Toggle

        groupSyncWrite.clearParam()

        for dxl_id in DXL_IDs:
            param = to_little_endian_bytes(goal_pos)
            groupSyncWrite.addParam(dxl_id, param)

        # Send position commands simultaneously
        groupSyncWrite.txPacket()

        # Timing variables for loop frequency measurement
        prev_time = time.time()
        count = 0
        freq_print_interval = 0.2  # seconds
        last_freq_print = prev_time

        # Wait until all servos reach position
        while True:
            dxl_comm_result = groupSyncRead.txRxPacket()
            if dxl_comm_result != COMM_SUCCESS:
                print(f"Error reading positions: {packetHandler.getTxRxResult(dxl_comm_result)}")
                break

            positions_reached = 0
            # Removed per-iteration print to avoid slowing down loop
            for dxl_id in DXL_IDs:
                current_pos = groupSyncRead.getData(dxl_id, ADDR_PRESENT_POSITION, 4)
                if abs(goal_pos - current_pos) < DXL_MOVING_STATUS_THRESHOLD:
                    positions_reached += 1

            # Calculate and print loop frequency once per second
            current_time = time.time()
            count += 1
            elapsed = current_time - prev_time
            if (current_time - last_freq_print) >= freq_print_interval:
                if elapsed > 0:
                    freq = count / elapsed
                    print(f"Loop frequency: {freq:.2f} Hz")
                prev_time = current_time
                count = 0
                last_freq_print = current_time

            if positions_reached == len(DXL_IDs):
                break

            time.sleep(0.01)  # ~100 Hz

        time.sleep(1)

except KeyboardInterrupt:
    print("🛑 Interrupted by user")

# Disable torque on all servos and close port
for dxl_id in DXL_IDs:
    packetHandler.write1ByteTxRx(portHandler, dxl_id, ADDR_TORQUE_ENABLE, TORQUE_DISABLE)
portHandler.closePort()
