import sys
import os
import time
import csv
import serial
from servo.servo_controller import ServoController
from statistics import mean
from imu.simple_monitor import load_calibration

# === Parameters ===
PORT_IMU = "COM8"
SERVO_STEP_SIZE = 10        # Steps between each position
WAIT_TIME = 0.6             # Seconds to wait for table to settle
SAMPLES = 5                 # Number of IMU samples per position

SERVO_MIN = 1500
SERVO_MAX = 2500

CALIBRATION_FILES = {
    0: "servo_calib_pitch.csv",
    1: "servo_calib_roll.csv"
}


def read_imu_pitch(ser, pitch_offset):
    """Read a single pitch value from the IMU, apply calibration offset"""
    while True:
        try:
            line = ser.readline().decode('utf-8', errors='ignore').strip()
            if line.startswith("DATA:"):
                heading, pitch, roll = map(float, line[5:].split(','))
                if not (heading == 0.0 and pitch == 0.0 and roll == 0.0):
                    return pitch - pitch_offset
        except Exception:
            continue


def read_imu_roll(ser, roll_offset):
    """Read a single roll value from the IMU, apply calibration offset"""
    while True:
        try:
            line = ser.readline().decode('utf-8', errors='ignore').strip()
            if line.startswith("DATA:"):
                heading, pitch, roll = map(float, line[5:].split(','))
                if not (heading == 0.0 and pitch == 0.0 and roll == 0.0):
                    return roll - roll_offset
        except Exception:
            continue


def flush_imu_buffer(ser):
    """Clear out old IMU lines from the buffer"""
    ser.reset_input_buffer()
    ser.reset_output_buffer()
    time.sleep(0.1)
    while ser.in_waiting:
        try:
            ser.readline()
        except:
            break


def calibrate_servo_axis(servo, ser, pitch_offset, roll_offset, axis_index):
    """Calibrate either pitch (0) or roll (1) servo and return results"""
    axis_name = "pitch" if axis_index == 0 else "roll"
    print(f"\n🧪 Starting {axis_name.upper()} calibration sweep from {SERVO_MIN} to {SERVO_MAX}...")

    results = []

    for pos in range(SERVO_MAX, SERVO_MIN - 1, -SERVO_STEP_SIZE):
        print(f"\n➡️  Setting servo[{axis_index}] to {pos}")
        
        center = servo.DXL_CENTER_POSITIONS[axis_index]
        angle_rad = (pos - center) / servo.STEPS_PER_RADIAN
        if axis_index == 0:
            servo.set_table_angles(pitch_rad=angle_rad, roll_rad=0)
        else:
            servo.set_table_angles(pitch_rad=0, roll_rad=-angle_rad)

        time.sleep(WAIT_TIME)
        flush_imu_buffer(ser)

        # Sample IMU readings
        samples = []
        for _ in range(SAMPLES):
            if axis_index == 0:
                value = read_imu_pitch(ser, pitch_offset)
            else:
                value = read_imu_roll(ser, roll_offset)
            print(f"   ↳ Sampled {axis_name}: {value:.2f}°")
            samples.append(value)
            time.sleep(0.05)

        avg_value = mean(samples)
        print(f"📈 IMU {axis_name} = {avg_value:.2f}°")
        results.append((pos, avg_value))

    return results


def main():
    # === Load IMU Calibration ===
    pitch_offset, roll_offset = load_calibration("imu/embedded_imu_calibration.txt")

    # === Setup IMU Serial ===
    print(f"📡 Connecting to IMU on {PORT_IMU}...")
    ser = serial.Serial(PORT_IMU, 115200, timeout=1)
    time.sleep(2)
    ser.reset_input_buffer()
    ser.reset_output_buffer()
    print("✅ IMU ready")

    # === Setup Servo Controller ===
    servo = ServoController()
    if not servo.connect():
        print("❌ Failed to connect to servo")
        return
    
    servo.set_table_angles(0, 0)
    time.sleep(2)

    for axis_index in [0, 1]:
        # Calibrate this axis
        results = calibrate_servo_axis(servo, ser, pitch_offset, roll_offset, axis_index)

        # Save calibration
        filename = CALIBRATION_FILES[axis_index]
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["ServoPosition", "Degrees"])
            writer.writerows(results)
        print(f"✅ Saved {filename}")

        # Reset table
        servo.set_table_angles(0, 0)
        time.sleep(1)

    # === Cleanup ===
    servo.disconnect()
    ser.close()
    print("\n🎉 Calibration complete for both pitch and roll.")


if __name__ == "__main__":
    main()
