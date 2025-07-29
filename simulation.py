import pybullet as p
import pybullet_data
import time
import numpy as np
from pid_controller import PIDController

p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)

plane_id = p.loadURDF("plane.urdf")

base_id = p.createMultiBody(
    baseMass=0,
    baseCollisionShapeIndex=p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.05, 0.05, 0.03]),
    baseVisualShapeIndex=p.createVisualShape(p.GEOM_BOX, halfExtents=[0.05, 0.05, 0.03], rgbaColor=[0.5, 0.5, 0.5, 1]),
    basePosition=[0, 0, 0.02],
)

table_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.25, 0.25, 0.004])
table_visual = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.25, 0.25, 0.004], rgbaColor=[0.0, 0.0, 0.0, 1])
table_mass = 1.0
table_start_pos = [0, 0, 0.06]
table_id = p.createMultiBody(table_mass, table_shape, table_visual, table_start_pos)

ball_radius = 0.02
ball_start_pos = [0.12, 0.15, 0.5]  # You can change this to drop the ball anywhere
ball_id = p.createMultiBody(
    baseMass=0.1,
    baseCollisionShapeIndex=p.createCollisionShape(p.GEOM_SPHERE, radius=ball_radius),
    baseVisualShapeIndex=p.createVisualShape(p.GEOM_SPHERE, radius=ball_radius, rgbaColor=[1, 0, 0, 1]),
    basePosition=ball_start_pos
)

# PID controllers for pitch (controls ball Y position) and roll (controls ball X position)
pitch_pid = PIDController(kp=10.0, ki=0.1, kd=2.0, output_limits=(-0.05, 0.05))
roll_pid = PIDController(kp=10.0, ki=0.1, kd=2.0, output_limits=(-0.05, 0.05))

dt = 1. / 240.

while True:
    # Get ball position
    ball_pos, _ = p.getBasePositionAndOrientation(ball_id)
    ball_x, ball_y, _ = ball_pos

    # Compute control outputs to bring ball back to (0,0)
    pitch_angle = -pitch_pid.update(ball_y, dt)
    #pitch_angle = 0
    roll_angle = roll_pid.update(ball_x, dt)
    #roll_angle=0

    # Update table orientation
    quat = p.getQuaternionFromEuler([pitch_angle, roll_angle, 0])
    p.resetBasePositionAndOrientation(table_id, table_start_pos, quat)

    events = p.getMouseEvents()
    # for e in events:
    #     # e[0] = event type (2 = mouse button)
    #     # e[3] = button state (1 = pressed, 0 = released)
    #     # e[4] = mouse button (0 = left, 1 = right, 2 = middle)
    #     if e[0] == 2 and e[4] == 3:# Left mouse button pressed
    #         print("Disturbance triggered!")
    #         # Apply an impulse to the ball at its current position, e.g. a little nudge in x/y
    #         impulse = [10, 0, 0]  # tweak values for strength & direction
    #         position = p.getBasePositionAndOrientation(ball_id)[0]
    #         p.applyExternalForce(ball_id, -1, impulse, position, p.WORLD_FRAME)

    p.stepSimulation()
    time.sleep(dt)
