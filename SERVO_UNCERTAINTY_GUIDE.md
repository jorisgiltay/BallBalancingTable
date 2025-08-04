# Servo Uncertainty Guide

## Overview

Servo uncertainty has been added to both the PID simulation (`compare_control.py`) and RL training environment (`reinforcement_learning/ball_balance_env.py`) to simulate realistic XL430-250T servo behavior.

## XL430-250T Servo Characteristics

The uncertainty model is tuned for your Dynamixel XL430-250T servos:

- **Update Rate**: 60Hz (16.7ms response time)
- **Backlash**: ±0.05° (minimal due to quality gears)
- **Position Noise**: ±0.02° (high resolution)
- **Hysteresis**: 0.01° (very low)
- **Compliance**: 0.5% (stiff servo, minimal deflection under load)

## PID Simulation Usage

### Running with Servo Uncertainty

```bash
# Default - servo uncertainty enabled
python compare_control.py --control-method pid --enable-visuals

# Toggle uncertainty during simulation by pressing 'u'
```

### Keyboard Controls

- `u` - Toggle servo uncertainty on/off during simulation
- `r` - Reset ball
- `p` - Switch to PID control
- `l` - Switch to RL control
- `q` - Quit

### Visual Feedback

The dashboard shows:
- **Servo Uncertainty status** in the status text
- **Commanded vs Actual angles** with error display
- **Real-time uncertainty parameters** when enabled

## RL Training Usage

### Training with Servo Uncertainty (Recommended)

```bash
# Train with realistic servo behavior (default)
cd reinforcement_learning
python train_rl.py --mode train --freq 50

# Train with perfect control (debugging only)
python train_rl.py --mode train --freq 50 --no-servo-uncertainty
```

### Testing Trained Models

```bash
# Test with same uncertainty as training
python train_rl.py --mode test --model ./models/best_model --freq 50

# Test with perfect control to see pure RL performance
python train_rl.py --mode test --model ./models/best_model --freq 50 --no-servo-uncertainty
```

## Benefits of Servo Uncertainty

### For RL Training

1. **Sim-to-Real Transfer**: Models trained with uncertainty work better on real hardware
2. **Robustness**: Agents learn to handle mechanical imperfections
3. **Generalization**: Better performance across different servo conditions
4. **Realistic Performance**: Training metrics match real-world expectations

### For PID Tuning

1. **Real-world Validation**: Test PID parameters against realistic servo behavior
2. **Cascaded Control Testing**: See how IMU feedback compensates for servo uncertainty
3. **System Identification**: Understand how mechanical imperfections affect control

## Implementation Details

### Uncertainty Components

1. **Response Delay**: 1 step delay (20ms at 50Hz) simulating 60Hz servo update rate
2. **Backlash**: ±0.05° dead zone requiring minimum movement to overcome
3. **Position Noise**: 0.02° standard deviation random jitter
4. **Saturation**: Smooth saturation near ±3.2° servo limits
5. **Hysteresis**: 0.01° direction-dependent offset
6. **Compliance**: 0.5% position error under table weight load

### Servo State Tracking

The system tracks:
- Commanded vs actual positions
- Delay buffers for response lag
- Movement direction for hysteresis
- Previous positions for backlash calculation

## Parameter Tuning

If your servo behavior differs, adjust parameters in the servo_uncertainty dictionary:

```python
self.servo_uncertainty = {
    'backlash_degrees': 0.05,     # Adjust for your servo's mechanical slack
    'response_delay_steps': 1,     # Adjust for your control frequency
    'position_noise_std': 0.02,   # Adjust for your servo's precision
    'saturation_softness': 0.9,   # Adjust saturation curve
    'hysteresis_strength': 0.01,  # Adjust directional bias
    'compliance_factor': 0.005,   # Adjust stiffness under load
}
```

## Comparison Results

You should see:
- **With Uncertainty OFF**: Unrealistically smooth control, possible oscillations when transferred to real hardware
- **With Uncertainty ON**: More realistic behavior, better sim-to-real transfer, slightly more challenging control

## Recommendations

1. **Always train RL with uncertainty enabled** for realistic performance
2. **Use uncertainty for final PID validation** before deploying to hardware
3. **Toggle uncertainty during development** to debug control algorithms
4. **Compare performance with/without** to understand uncertainty impact

The servo uncertainty model makes your simulation much more representative of real XL430-250T behavior, leading to better control performance when deployed to actual hardware.
