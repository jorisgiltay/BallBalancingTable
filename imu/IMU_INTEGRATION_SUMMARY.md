# IMU Integration with Ball Balancing Control System

## 🎯 Overview

The IMU feedback has been successfully integrated into the `compare_control.py` script, allowing real-time comparison between simulation angles and actual hardware feedback from the BNO055 IMU sensor.

## ✅ Integration Features

### 🧭 IMU Interface
- **Background Threading**: Continuous 100Hz IMU data reading without blocking main control loop
- **Thread-Safe Data Sharing**: Atomic updates of IMU readings accessible to main simulation
- **Connection Management**: Robust connection handling with error recovery
- **Hot-Pluggable**: IMU can be enabled/disabled via command line arguments

### 📊 Real-Time Monitoring
- **Live Angle Comparison**: Shows difference between simulation table angles and real IMU readings
- **Visual Dashboard Integration**: IMU data appears in the matplotlib dashboard
- **Console Status Display**: Periodic status updates include IMU pitch/roll readings
- **Angle Difference Calculation**: Automatic computation of simulation vs reality differences

### 🔧 Command Line Integration
- `--imu`: Enable IMU feedback
- `--imu-port COM6`: Specify COM port (default: COM6)
- Compatible with all existing modes (visuals, servos, camera)

## 🚀 Usage Examples

### Basic IMU Monitoring
```bash
python compare_control.py --imu
```

### IMU + Visual Dashboard
```bash
python compare_control.py --imu --visuals
```

### Full Hardware Integration
```bash
python compare_control.py --imu --servos --camera real
```

### Hybrid Mode (Camera + IMU)
```bash
python compare_control.py --imu --camera hybrid --visuals
```

## 📋 Technical Implementation

### Code Architecture
1. **Optional Import Pattern**: Same pattern as servo/camera integration
2. **Background Thread**: `_imu_reader_thread()` runs continuously at 100Hz
3. **Data Parsing**: Parses "DATA: heading,pitch,roll" from Arduino
4. **Thread-Safe Updates**: Atomic assignment of pitch/roll/heading values
5. **Integration Points**: IMU data flows into visual system and status display

### Key Files Modified
- **`compare_control.py`**: Main integration with IMU support
- **`imu/imu_simple.py`**: Simple IMU interface (unchanged, used as-is)
- **`test_imu_integration.py`**: Integration test script
- **`demo_imu_integration.py`**: Demonstration of IMU features

### Integration Points
```python
# IMU feedback in simulation loop
imu_feedback = self.get_imu_feedback()

# Visual dashboard shows IMU vs simulation differences
imu_pitch_diff = data['imu']['pitch'] - np.degrees(data['pitch'])
imu_roll_diff = data['imu']['roll'] - np.degrees(data['roll'])

# Console status includes IMU readings
if imu_feedback['connected']:
    status_line += f" | IMU: P{imu_feedback['pitch']:+.1f}° R{imu_feedback['roll']:+.1f}°"
```

## 🎯 Benefits

### For Development
- **Validation**: Compare simulation physics with real hardware behavior
- **Debugging**: Identify discrepancies between expected and actual table angles
- **Calibration**: Fine-tune PID parameters based on real-world feedback
- **Testing**: Validate control algorithms before full hardware deployment

### For Operation
- **Real-Time Monitoring**: Continuous feedback on actual table position
- **Performance Analysis**: Track how well simulation matches reality
- **System Health**: Monitor IMU connection and data quality
- **Hybrid Operation**: Mix simulation convenience with hardware validation

## 📈 Performance Characteristics

- **Data Rate**: 100Hz IMU readings (same as Arduino output)
- **Display Rate**: Every 10th reading shown to avoid overwhelming console
- **Thread Overhead**: Minimal - dedicated IMU thread doesn't block control loop
- **Memory Usage**: Very low - only stores latest IMU readings
- **CPU Impact**: Negligible additional load

## 🔄 Workflow Integration

### Development Workflow
1. **Pure Simulation**: Test algorithms in PyBullet (`--control pid/rl`)
2. **IMU Validation**: Add IMU to compare simulation vs hardware (`--imu --visuals`)
3. **Hardware Testing**: Full integration with servos (`--imu --servos`)
4. **Complete System**: All hardware active (`--imu --servos --camera real`)

### Real-Time Feedback Loop
```
Arduino BNO055 → Serial (100Hz) → IMU Thread → Main Control Loop
    ↓                                              ↓
 IMU Data                                    Servo Commands
    ↓                                              ↓
Visual Dashboard ← Status Display ← Angle Comparison
```

## 🧪 Testing & Validation

### Test Scripts Available
- **`test_imu_integration.py`**: Verify integration works correctly
- **`demo_imu_integration.py`**: Show mock IMU data flow
- **`imu/test_imu_simple.py`**: Test raw IMU interface

### Validation Steps
1. ✅ IMU import and initialization
2. ✅ Background thread functionality  
3. ✅ Thread-safe data sharing
4. ✅ Visual dashboard integration
5. ✅ Command line argument handling
6. ✅ Graceful cleanup on exit

## 🎮 Control Flow

The IMU integration follows this flow in the main simulation loop:

```python
while True:
    # Control logic (every control_freq Hz)
    if physics_step_count % physics_steps_per_control == 0:
        # Get ball observation
        observation = self.get_observation()
        
        # Run control algorithm (PID/RL)
        control_action = self.control_algorithm(observation)
        
        # Update table angles
        self.table_pitch, self.table_roll = control_action
        
        # Send to servos (if enabled)
        if self.servo_controller:
            self.servo_controller.set_table_angles(self.table_pitch, self.table_roll)
        
        # Get IMU feedback for comparison
        imu_feedback = self.get_imu_feedback()  # ← NEW: IMU integration
        
        # Update visuals with IMU data
        self._update_visual_data(..., imu_feedback)  # ← NEW: IMU in visuals
        
        # Status display with IMU
        print(f"Status | IMU: P{imu_feedback['pitch']:+.1f}°")  # ← NEW: IMU status
    
    # Physics simulation continues at physics_freq Hz
    p.stepSimulation()
```

## 🎉 Ready for Use!

The IMU integration is now complete and ready for ball balancing control! Your system can now:

- **Monitor real-time table angles** from the BNO055 IMU
- **Compare simulation vs reality** to validate control algorithms  
- **Display angle differences** in both console and visual dashboard
- **Operate in pure simulation or mixed hardware modes**
- **Provide continuous feedback** for control system development

Connect your Arduino with BNO055, upload the `arduino_bno055_simple.ino` sketch, and run:

```bash
python compare_control.py --imu --visuals
```

Watch your ball balancing system with real-time IMU feedback! 🎯🧭
