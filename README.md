# Ball Balancing Table

A complete ball balancing control system with **simulation**, **camera integration**, and **hardware control**. Features PID and Reinforcement Learning controllers with real-time visual monitoring and servo control for physical deployment.

![PyBullet Simulation](media/sim.png)

## 🎯 Features

### Unified Control System
- **🎛️ PID Control**: Traditional control with tuned parameters
- **🤖 Reinforcement Learning**: PPO agent with advanced reward engineering  
- **📷 Camera Integration**: RealSense camera for real-world ball tracking
- **🦾 Servo Control**: Dynamixel servo integration for hardware deployment
- **⚡ Real-time Switching**: Change control methods during operation

### Operating Modes
- **🖥️ Pure Simulation**: PyBullet physics simulation only
- **🔗 Hybrid Mode**: Camera input + simulated physics for testing
- **🏗️ Hardware Mode**: Full camera + servo deployment

![Hybrid Mode Demo](media/hybrid_mode.gif)

### Visual Dashboard
- **📊 Real-time Monitoring**: Live ball position, control actions, and performance metrics
- **🎨 Professional Interface**: Clean, dark-themed dashboard with color-coded status
- **📈 Performance Analysis**: Distance tracking, velocity monitoring, and control efficiency

![PID Control Dashboard](media/matplotlib_PID.png)
![RL Control Dashboard](media/matplotlib_RL.png)

## 🚀 Quick Start

### 1. Installation
```bash
# Install dependencies
pip install -r requirements.txt

# Setup project
python setup.py
```

### 2. Basic Simulation
```bash
# Pure simulation with PID control
python compare_control.py --control pid --visuals

# Test RL control (if model available)
python compare_control.py --control rl --visuals
```

### 3. Camera Integration
```bash
# Calibrate camera first (requires RealSense + blue markers)
python compare_control.py --camera hybrid --calibrate

# Run hybrid mode with camera input
python compare_control.py --camera hybrid --visuals
```

### 4. Hardware Deployment
```bash
# Full hardware mode with servos
python compare_control.py --camera real --servos --calibrate
```

## 🎮 Interactive Controls

**Keyboard shortcuts during operation:**
- `r` - Reset ball position
- `f` - Toggle fixed/random ball positions  
- `p` - Switch to PID control
- `l` - Switch to RL control
- `q` - Quit

## ⚙️ Configuration Options

### Camera Modes
- `--camera simulation` - Pure PyBullet simulation (default)
- `--camera hybrid` - Camera input + simulated physics  
- `--camera real` - Camera only for hardware deployment

### Control Options
- `--control pid` - Start with PID controller (default)
- `--control rl` - Start with RL controller
- `--freq 50` - Control frequency in Hz (default: 50)

### Hardware Integration
- `--servos` - Enable Dynamixel servo control
- `--calibrate` - Run camera calibration before starting
- `--visuals` - Enable real-time dashboard

### Example Commands
```bash
# Development: Hybrid mode with visuals
python compare_control.py --camera hybrid --visuals --servos

# Hardware: Full deployment
python compare_control.py --camera real --servos --calibrate

# Simulation: Performance testing  
python compare_control.py --control rl --freq 100 --visuals
```

## 🔧 Hardware Setup

### Camera System
- **Intel RealSense D435i** (or compatible)
- **35×35cm wooden base plate** with 4 blue corner markers
- **Camera positioned above table** for full table view

### Servo System  
- **2× Dynamixel servos** (XM430-W350 or similar)
- **USB interface** (U2D2 or compatible)
- **Kinematic model**: Half servo range = 3° table movement

### Calibration Requirements
- 4× **4cm blue markers** at base plate corners
- **Good lighting** for consistent marker detection
- **No ball on table** during calibration

## 📁 Project Structure

```
├── compare_control.py         # 🎯 Main unified control system
├── servo_controller.py        # 🦾 Servo control with kinematics  
├── camera_interface.py        # 📷 Camera integration & ball detection
├── camera_calibration_color.py # 🎯 Camera calibration tool
├── pid_controller.py          # 🎛️ PID controller implementation
├── ball_balance_env.py        # 🏋️ RL training environment
├── train_rl.py               # 🤖 RL training script
├── requirements.txt          # 📦 Dependencies
├── calibration_data/         # 📐 Camera calibration files
├── models/                   # 🧠 Trained RL models
├── good_models/              # ✅ Best model backups
└── media/                    # 📸 Documentation images
```

## 🤖 Reinforcement Learning

### Training New Models
```bash
# Train RL agent
python train_rl.py --mode train --freq 50

# Monitor training progress
tensorboard --logdir=./tensorboard_logs/
```

### Environment Details
- **Observation**: Ball position (x,y), velocity (vx,vy), table angles (pitch,roll)
- **Actions**: Table angle changes (±0.05 rad)  
- **Reward**: Distance minimization + energy optimization + oscillation prevention
- **Physics**: 25cm table, 2.7g ball, realistic dynamics

### Training Results

The PPO training shows excellent convergence with the improved reward function:

![TensorBoard Training Results](media/tensorboard_ppo_results.png)

**Key Training Metrics:**
- **Episode Reward Mean**: Steady improvement from -100 to optimal performance
- **Evaluation Mean Reward**: Consistent high performance during evaluation phases
- **Learning Rate**: Adaptive scheduling for stable convergence
- **Training Steps**: 330,000+ steps with checkpoints every 10k steps

The training demonstrates the effectiveness of our advanced reward engineering, with the agent learning to avoid bang-bang oscillations and achieving smooth, efficient control.

## 📷 Camera Calibration

The system uses blue corner markers for camera-to-table coordinate transformation:

```bash
# Interactive calibration (recommended)
python compare_control.py --camera hybrid --calibrate
# Choose option 1 for interactive calibration

# Or run calibration separately
python camera_calibration_color.py
```

**Calibration Process:**
1. Setup 35×35cm base plate with 4 blue markers at corners
2. Position RealSense camera above table
3. Run calibration to capture marker positions
4. System automatically calculates coordinate transformation

## 🦾 Servo Integration

### Hardware Configuration
- **Servo IDs**: 1 (pitch), 2 (roll)
- **Communication**: COM5, 1Mbps (configurable)
- **Range**: ±3° table movement for half servo range
- **Protocol**: Dynamixel Protocol 2.0

### Kinematic Model
```python
# Table angle to servo position conversion
STEPS_PER_RADIAN = 29,325  # Approximately
servo_position = center_position + (angle_rad * STEPS_PER_RADIAN)
```

### Testing Servos
```bash
# Test servo functionality
python servo_controller.py
```

## 🔬 Advanced Features

### Multi-Mode Operation
- Seamlessly switch between simulation, hybrid, and hardware modes
- Camera calibration integrates with existing simulation
- Servo control mirrors simulation movements

### Safety Features
- **Angle Limits**: Software limits prevent servo damage
- **Connection Monitoring**: Automatic fallback if hardware disconnects  
- **Calibration Validation**: Ensures reliable camera-table mapping

### Performance Optimization
- **50Hz Control Loop**: Matches servo update rates
- **Thread-safe Visuals**: Non-blocking dashboard updates
- **Efficient Communication**: Optimized servo commands and camera processing

## 🚀 Development Workflow

### 1. Algorithm Development
```bash
# Develop and test in pure simulation
python compare_control.py --control pid --visuals
```

### 2. Camera Integration Testing  
```bash
# Test camera integration with hybrid mode
python compare_control.py --camera hybrid --visuals
```

### 3. Hardware Deployment
```bash
# Deploy to real hardware
python compare_control.py --camera real --servos --calibrate
```

### 4. Performance Analysis
```bash
# Compare control methods with full instrumentation
python compare_control.py --camera hybrid --servos --visuals
```

## 💡 Tips & Troubleshooting

### Camera Issues
- Ensure RealSense drivers are installed
- Check USB 3.0 connection for camera
- Verify blue markers are clearly visible and unobstructed
- Re-run calibration if ball tracking seems inaccurate

### Servo Issues  
- Check COM port and baudrate settings
- Verify Dynamixel power supply and connections
- Test individual servos with manufacturer tools
- Ensure servo IDs match configuration (1 for pitch, 2 for roll)

### Performance Optimization
- Use `--freq` to adjust control frequency based on hardware capabilities
- Disable `--visuals` for maximum performance in deployment
- Monitor CPU usage during camera processing

## 📋 System Requirements

- **Python 3.8+**
- **PyBullet** - Physics simulation
- **OpenCV** - Camera processing  
- **pyrealsense2** - RealSense camera support
- **dynamixel_sdk** - Servo control
- **stable-baselines3** - RL algorithms
- **matplotlib** - Real-time visualization

**Hardware Requirements:**
- Intel RealSense D435i camera
- 2× Dynamixel servos (XM430-W350 recommended)
- USB-Serial interface for servos
- 35×35cm base plate with blue corner markers
