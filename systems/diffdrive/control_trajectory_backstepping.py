import os
import sys
wd = os.path.abspath(os.getcwd())
sys.path.append(str(wd))

import numpy as np
import matplotlib.pyplot as plt
from robot.mobile.differential import DifferentialDrive
from controllers.differential.trajectory_backstepping import DifferentialDriveBackSteppingTrajectoryController
from simulator.integrator_euler import EulerNumericalIntegrator
from trajectory_generator.traj_primitive import circle

# create robot and controller
robot = DifferentialDrive(wheelRadius=0.03, baseLength=0.3, baseWidth=1)
controller = DifferentialDriveBackSteppingTrajectoryController(robot=robot)

# simulator
def dynamic(currentPose, input):
    return robot.forward_external_kinematic(input, currentPose[2,0])

def desired(currentPose, time):
    xRef, yRef, vr, wr, ydot, xdot, vdotref, wdotref = circle(time)
    return np.array([[xRef], [yRef], [vr], [wr], [xdot], [ydot]])

def control(currentPose, desiredPose):
    xRef = desiredPose[0,0]
    yRef = desiredPose[1,0]
    vr = desiredPose[2,0]
    wr = desiredPose[3,0]
    xdot = desiredPose[4,0]
    ydot = desiredPose[5,0]
    thetaRef = np.arctan2(ydot, xdot)
    qr = np.array([[xRef], [yRef], [thetaRef]])
    return controller.kinematic_control(currentPose, qr, vr, wr)

q0 = np.array([[6], [0], [0]])
tSpan = (0, 50)
dt = 0.01
intg = EulerNumericalIntegrator(dynamic, control, desired, q0, tSpan, dt)
timeSteps, states, desireds, controls = intg.simulation()

plt.plot(states[0,:], states[1,:])
# plt.plot(timeSteps, states[0,:])
# plt.plot(timeSteps, states[1,:])
# plt.plot(timeSteps, states[2,:])
plt.show()