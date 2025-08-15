import argparse
import math
import sys
import time
from statistics import mean

import numpy as np

# Support running as a module or a script
try:
    from servo_controller import ServoController
except Exception:
    from servo_controller import ServoController
try:
    from dynamixel_sdk import GroupSyncRead
except Exception:
    GroupSyncRead = None


def clamp(value: float, min_value: float, max_value: float) -> float:
    return max(min_value, min(max_value, value))


def run_best_effort(controller: ServoController, duration_s: float, oscillate: bool, amplitude_deg: float, wave_hz: float) -> dict:
    t0 = time.perf_counter()
    last = t0
    count = 0
    samples = []
    while True:
        now = time.perf_counter()
        if now - t0 >= duration_s:
            break

        if oscillate and amplitude_deg > 0.0:
            phase = 2.0 * math.pi * wave_hz * (now - t0)
            pitch = math.radians(amplitude_deg) * math.sin(phase)
            roll = math.radians(amplitude_deg) * math.cos(phase)
        else:
            pitch = 0.0
            roll = 0.0

        controller.set_table_angles(pitch, roll)

        count += 1
        samples.append(now - last)
        last = now

    total_time = max(1e-6, time.perf_counter() - t0)
    hz = count / total_time
    if samples:
        arr = np.array(samples)
        stats = {
            'avg_dt_ms': float(arr.mean() * 1000.0),
            'p50_dt_ms': float(np.percentile(arr, 50) * 1000.0),
            'p90_dt_ms': float(np.percentile(arr, 90) * 1000.0),
            'p99_dt_ms': float(np.percentile(arr, 99) * 1000.0),
        }
    else:
        stats = {'avg_dt_ms': 0.0, 'p50_dt_ms': 0.0, 'p90_dt_ms': 0.0, 'p99_dt_ms': 0.0}

    return {'count': count, 'seconds': total_time, 'hz': hz, **stats}


def run_target_rate(controller: ServoController, duration_s: float, target_hz: float, oscillate: bool, amplitude_deg: float, wave_hz: float) -> dict:
    dt = 1.0 / max(1.0, target_hz)
    t0 = time.perf_counter()
    next_t = t0
    count = 0
    samples = []
    while True:
        now = time.perf_counter()
        if now - t0 >= duration_s:
            break

        if oscillate and amplitude_deg > 0.0:
            phase = 2.0 * math.pi * wave_hz * (now - t0)
            pitch = math.radians(amplitude_deg) * math.sin(phase)
            roll = math.radians(amplitude_deg) * math.cos(phase)
        else:
            pitch = 0.0
            roll = 0.0

        iter_start = time.perf_counter()
        controller.set_table_angles(pitch, roll)
        iter_elapsed = time.perf_counter() - iter_start
        samples.append(iter_elapsed)

        count += 1
        next_t += dt
        sleep_time = next_t - time.perf_counter()
        if sleep_time > 0:
            time.sleep(sleep_time)
        else:
            next_t = time.perf_counter()

    total_time = max(1e-6, time.perf_counter() - t0)
    hz = count / total_time
    if samples:
        arr = np.array(samples)
        stats = {
            'avg_send_ms': float(arr.mean() * 1000.0),
            'p50_send_ms': float(np.percentile(arr, 50) * 1000.0),
            'p90_send_ms': float(np.percentile(arr, 90) * 1000.0),
            'p99_send_ms': float(np.percentile(arr, 99) * 1000.0),
        }
    else:
        stats = {'avg_send_ms': 0.0, 'p50_send_ms': 0.0, 'p90_send_ms': 0.0, 'p99_send_ms': 0.0}

    return {'count': count, 'seconds': total_time, 'hz': hz, **stats}


def run_verified_rate(controller: ServoController, duration_s: float, target_hz: float, oscillate: bool, amplitude_deg: float, wave_hz: float) -> dict:
    """Write goal, then read back GOAL_POSITION for all servos each cycle to verify acceptance.

    This measures an end-to-end command+verify cycle frequency. It will be lower than write-only.
    """
    if GroupSyncRead is None or controller.port_handler is None or controller.packet_handler is None:
        raise RuntimeError("dynamixel_sdk GroupSyncRead not available or controller not connected")

    # Sync read GOAL_POSITION (addr 116, length 4) for XM430 (Prot 2.0)
    GOAL_POS_ADDR = 116
    GOAL_POS_LEN = 4
    sync_read = GroupSyncRead(controller.port_handler, controller.packet_handler, GOAL_POS_ADDR, GOAL_POS_LEN)
    for sid in controller.servo_ids:
        sync_read.addParam(sid)

    dt = 1.0 / max(1.0, target_hz)
    t0 = time.perf_counter()
    next_t = t0
    count = 0
    mismatches = 0

    while True:
        now = time.perf_counter()
        if now - t0 >= duration_s:
            break

        # Compute desired pitch/roll (small safe oscillation or zero)
        if oscillate and amplitude_deg > 0.0:
            phase = 2.0 * math.pi * wave_hz * (now - t0)
            pitch = math.radians(amplitude_deg) * math.sin(phase)
            roll = math.radians(amplitude_deg) * math.cos(phase)
        else:
            pitch = 0.0
            roll = 0.0

        # Compute expected servo positions (replicate controller's mapping)
        pitch_cmd = (controller.pitch_gain * pitch) + (controller.c_pr * roll) + controller.pitch_offset_rad
        roll_cmd = (controller.roll_gain * roll) + (controller.c_rp * pitch) + controller.roll_offset_rad
        expected_pitch_pos = controller.angle_to_servo_position(pitch_cmd, 0)
        expected_roll_pos = controller.angle_to_servo_position(-roll_cmd, 1)

        # Write commands
        controller.set_table_angles(pitch, roll)

        # Read back goal positions for verification
        if sync_read.txRxPacket() != 0:
            # Communication error; count as mismatch and continue
            mismatches += 1
        else:
            try:
                # IDs assumed in order [pitch_id, roll_id]
                gp_pitch = sync_read.getData(controller.servo_ids[0], GOAL_POS_ADDR, GOAL_POS_LEN)
                gp_roll = sync_read.getData(controller.servo_ids[1], GOAL_POS_ADDR, GOAL_POS_LEN)
                if gp_pitch != expected_pitch_pos or gp_roll != expected_roll_pos:
                    mismatches += 1
            except Exception:
                mismatches += 1

        count += 1
        next_t += dt
        sleep_time = next_t - time.perf_counter()
        if sleep_time > 0:
            time.sleep(sleep_time)
        else:
            next_t = time.perf_counter()

    total_time = max(1e-6, time.perf_counter() - t0)
    hz = count / total_time
    return {
        'count': count,
        'seconds': total_time,
        'hz': hz,
        'mismatches': mismatches,
        'match_rate': 0.0 if count == 0 else (1.0 - (mismatches / count))
    }


def main():
    parser = argparse.ArgumentParser(description="Measure achievable servo control frequency safely")
    parser.add_argument('--device', default='COM5', help='Serial device name (default COM5)')
    parser.add_argument('--baud', type=int, default=1000000, help='Baudrate (default 1,000,000)')
    parser.add_argument('--duration', type=float, default=5.0, help='Test duration in seconds (default 5)')
    parser.add_argument('--mode', choices=['best', 'target', 'verify'], default='best', help='Test mode: best effort, target rate, or verified (write + readback)')
    parser.add_argument('--target-hz', type=float, default=100.0, help='Target Hz for target mode (default 100)')
    parser.add_argument('--oscillate', action='store_true', help='Oscillate with small safe amplitude instead of commanding zero')
    parser.add_argument('--amplitude-deg', type=float, default=0.5, help='Oscillation amplitude in degrees (default 0.5°, capped to 3°)')
    parser.add_argument('--wave-hz', type=float, default=1.0, help='Oscillation frequency in Hz (default 1)')

    args = parser.parse_args()

    safe_amplitude = clamp(args.amplitude_deg, 0.0, 3.0)

    controller = ServoController(device_name=args.device, baudrate=args.baud)
    if not controller.connect():
        print("Failed to connect to servos. Aborting.")
        sys.exit(1)

    try:
        controller.set_table_angles(0.0, 0.0)
        time.sleep(0.5)

        if args.mode == 'best':
            result = run_best_effort(controller, args.duration, args.oscillate, safe_amplitude, args.wave_hz)
            print(f"Mode: BEST-EFFORT | Duration: {result['seconds']:.2f}s | Cycles: {result['count']}")
            print(f"Achieved: {result['hz']:.1f} Hz | dt avg/p50/p90/p99: {result['avg_dt_ms']:.3f}/{result['p50_dt_ms']:.3f}/{result['p90_dt_ms']:.3f}/{result['p99_dt_ms']:.3f} ms")
        elif args.mode == 'target':
            result = run_target_rate(controller, args.duration, args.target_hz, args.oscillate, safe_amplitude, args.wave_hz)
            print(f"Mode: TARGET {args.target_hz:.1f} Hz | Duration: {result['seconds']:.2f}s | Cycles: {result['count']}")
            print(f"Achieved: {result['hz']:.1f} Hz | send ms avg/p50/p90/p99: {result['avg_send_ms']:.3f}/{result['p50_send_ms']:.3f}/{result['p90_send_ms']:.3f}/{result['p99_send_ms']:.3f}")
        else:
            result = run_verified_rate(controller, args.duration, args.target_hz, args.oscillate, safe_amplitude, args.wave_hz)
            print(f"Mode: VERIFY (write+readback GOAL) {args.target_hz:.1f} Hz | Duration: {result['seconds']:.2f}s | Cycles: {result['count']}")
            print(f"Achieved: {result['hz']:.1f} Hz | Match rate: {result['match_rate']*100:.1f}% ({result['mismatches']} mismatches)")

    except KeyboardInterrupt:
        print("Interrupted by user")
    finally:
        controller.set_table_angles(0.0, 0.0)
        time.sleep(0.5)
        controller.disconnect()


if __name__ == '__main__':
    main()


