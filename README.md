# Ball Balancing Table

A comprehensive physics simulation of a ball balancing on a pivoting table, featuring both PID control and Reinforcement Learning approaches with real-time visual monitoring.

![PyBullet Simulation](media/sim.png)

## 🎯 Features

### Control Systems
- **🎛️ PID Control**: Traditional control system with tuned parameters for reliable performance
- **🤖 Reinforcement Learning**: PPO agent trained with advanced reward engineering
- **⚡ Real-time Comparison**: Switch between control methods during simulation
- **🔧 Live Tuning**: Adjust control parameters on-the-fly

### Visual Dashboard
- **📊 Matplotlib Dashboard**: Real-time performance monitoring in separate window
- **🎨 Professional Interface**: Clean, dark-themed dashboard with color-coded status
- **📈 Live Metrics**: Ball position, velocity, control actions, and table angles
- **🎯 Visual Feedback**: Color-coded performance indicators and trend monitoring

![PID Control Dashboard](media/matplotlib_PID.png)
![RL Control Dashboard](media/matplotlib_RL.png)

### Advanced Environment
- **🎯 Realistic Physics**: 25cm×25cm table with 2.7g ping pong ball (real-world dimensions)
- **⚙️ Professional Timing**: 240Hz physics, 50Hz control frequency (servo-realistic)
- **🎨 Clean Visuals**: Gray ground plane, dark metallic table, bright white ball
- **🔄 Thread-safe Architecture**: Non-blocking visual system with queue-based communication

## 🚀 Quick Start

### 1. Setup Environment
```bash
python setup.py
```

### 2. Test PID Control with Visual Dashboard
```bash
python compare_control.py --control pid --visuals
```

### 3. Train RL Agent (with improved reward function)
```bash
python train_rl.py --mode train --freq 50
```

### 4. Test RL Agent with Dashboard
```bash
python compare_control.py --control rl --visuals
```

### 5. Compare Both Methods
```bash
python compare_control.py --control pid --freq 50 --visuals
```
Then press `l` to switch to RL control in real-time!

## 🎮 Interactive Controls

During simulation, use these keyboard shortcuts:
- `r` - Reset ball position
- `f` - Toggle fixed/random ball starting positions
- `p` - Switch to PID control
- `l` - Switch to RL control (if model available)
- `q` - Quit simulation

## 📁 Project Structure

```
├── simulation.py          # Original PID-only simulation
├── pid_controller.py      # Tuned PID controller implementation
├── ball_balance_env.py    # Advanced Gymnasium environment for RL
├── train_rl.py           # RL training with improved reward engineering
├── compare_control.py    # Interactive comparison tool with visual dashboard
├── debug_rl.py           # RL debugging and analysis tools
├── recovery_tool.py      # Model recovery and checkpoint management
├── setup.py              # Setup and installation script
├── requirements.txt      # Python dependencies
├── media/                # Screenshots and documentation images
├── models/               # Trained RL models
├── good_models/          # Backup of best performing models
├── checkpoints/          # Training checkpoints every 10k steps
└── tensorboard_logs/     # Detailed training logs
```

## 🤖 Reinforcement Learning Details

### Environment Specifications
- **Observation Space**: 6D - Ball position (x,y), velocity (vx,vy), table angles (pitch,roll)
- **Action Space**: 2D - Changes to table pitch and roll angles (±0.05 rad)
- **Physics**: Realistic 25cm table, 2.7g ball, proper friction and inertia
- **Control Frequency**: 50Hz (servo-realistic timing)

### Advanced Reward Engineering
- **Position Dominance**: Primary reward for staying near center (3x weight)
- **Control Energy Function**: PD controller guidance for optimal actions
- **Bang-bang Penalty**: Sophisticated oscillation detection and prevention
- **Circular Motion Detection**: Prevents orbital patterns around center
- **Velocity Context**: Smart velocity penalties based on ball position

### Training Configuration
- **Algorithm**: PPO (Proximal Policy Optimization) from stable-baselines3
- **Training Steps**: 330,000+ steps with checkpoints every 10k
- **Evaluation**: Comprehensive eval every 10k steps
- **Monitoring**: TensorBoard logs with detailed metrics

Monitor training progress:
```bash
tensorboard --logdir=./tensorboard_logs/
```

### Training Results

The PPO training shows excellent convergence with the improved reward function:

![TensorBoard Training Results](media/tensorboard_ppo_results.png)

**Key Training Metrics:**
- **Episode Reward Mean**: Steady improvement from -100 to optimal performance
- **Evaluation Mean Reward**: Consistent high performance during evaluation phases  
- **Learning Rate**: Adaptive scheduling for stable convergence
- **Training Steps**: 330,000+ steps with checkpoints every 10k steps

The training demonstrates the effectiveness of our advanced reward engineering, with the agent learning to avoid bang-bang oscillations and achieving smooth, efficient control.

## 📊 Performance Monitoring

### Visual Dashboard Features
- **Ball Position Tracker**: Real-time top-down view with table boundary
- **Control Action Bars**: Live display of pitch/roll commands with magnitude indicators
- **Table Angle Monitor**: Current table orientation with safety limits
- **Performance Metrics**: Distance, velocity, action magnitude, and control method
- **Color-coded Status**: Green (excellent), Yellow (good), Red (poor) performance indicators

### Thread-safe Architecture
- **Non-blocking Updates**: Dashboard updates don't interfere with control timing
- **Queue-based Communication**: Smooth data flow between simulation and visualization
- **Optimized Rendering**: Blitting and selective updates for 60+ FPS dashboard performance

## 🎯 Hardware-Ready Design

### Real-world Compatibility
- **Servo Timing**: 50Hz control frequency matches standard servo update rates
- **Realistic Dimensions**: 25cm×25cm table matches typical hardware builds
- **Physical Properties**: 2.7g ping pong ball with realistic friction coefficients
- **Sensor Simulation**: Position estimation mimics camera/sensor input

### Control Theory Implementation
- **PD Controller Guidance**: RL reward function includes optimal action calculation
- **Bang-bang Prevention**: Advanced oscillation detection prevents instability
- **Energy Optimization**: Minimal control effort for maximum stability

## 🔬 Advanced Features

### Model Management
- **Automatic Checkpointing**: Models saved every 10,000 training steps
- **Best Model Tracking**: Automatic backup of highest-performing models
- **Recovery Tools**: Utilities for checkpoint analysis and model recovery

### Debugging and Analysis
- **RL Debugging**: Comprehensive tools for analyzing agent behavior
- **Performance Comparison**: Side-by-side PID vs RL evaluation
- **Visual Analysis**: Real-time monitoring of control decisions

## 🚀 Next Steps

### Immediate Improvements
1. **Hardware Integration**: Connect to actual servos and camera system
2. **Computer Vision**: Replace simulated position with real camera input
3. **Parameter Optimization**: Use the visual dashboard to fine-tune PID gains

### Advanced Development
1. **Multi-ball Scenarios**: Handle multiple balls simultaneously
2. **Disturbance Rejection**: Add external forces and vibrations
3. **Adaptive Control**: Online learning and parameter adjustment
4. **Curriculum Learning**: Progressive difficulty increase during training

### Research Directions
1. **Advanced RL Algorithms**: Experiment with SAC, TD3, or custom architectures
2. **Transfer Learning**: Train in simulation, deploy on hardware
3. **Robust Control**: Handle model uncertainties and real-world variations

## 💡 Tips for Development

### For RL Training
- Monitor the visual dashboard during training to understand agent behavior
- Use the comparison tool to validate RL performance against PID baseline
- Watch for bang-bang oscillations - the reward function should prevent them
- Training plateau around 200k steps is normal - the refined reward function continues improving

### For Hardware Deployment
- The 50Hz control frequency is optimized for standard servo systems
- Ball position estimation logic is designed for camera input conversion
- PID parameters are tuned for realistic response times and stability

### For Experimentation
- Use `--visuals` flag to understand what each control method is doing
- Press `f` to toggle between fixed and random starting positions
- The matplotlib dashboard provides insights into control effort and efficiency

## 📋 System Requirements

- Python 3.8+
- PyBullet for physics simulation
- stable-baselines3 for RL algorithms
- matplotlib for real-time visualization
- numpy, gymnasium for mathematical operations

**Performance Note**: The visual dashboard is optimized for smooth operation but can be disabled for maximum simulation speed by omitting the `--visuals` flag.
