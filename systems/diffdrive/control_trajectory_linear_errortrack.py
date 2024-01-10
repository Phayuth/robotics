import os
import sys
wd = os.path.abspath(os.getcwd())
sys.path.append(str(wd))

import numpy as np
import matplotlib.pyplot as plt
from robot.mobile.differential import DifferentialDrive
from controllers.differential.trajectory_linear_errortrack import DifferentialDriveErrorLinearTrajectoryController
from simulator.integrator_euler import EulerNumericalIntegrator
from trajectory_generator.traj_primitive import fig_of_8

# create robot and controller
robot = DifferentialDrive(wheelRadius=0.03, baseLength=0.3, baseWidth=0.3)
controller = DifferentialDriveErrorLinearTrajectoryController(robot=robot)

# simulator
def dynamic(currentPose, input):
    return robot.forward_external_kinematic(input, currentPose[2,0])

def desired(currentPose, time):
    xRef, yRef, dxRef, dyRef, ddxRef, ddyRef, vRef, wRef = fig_of_8(time)
    return np.array([[xRef], [yRef], [dxRef], [dyRef], [vRef], [wRef]])

def control(currentPose, desiredPose):
    xRef = desiredPose[0,0]
    yRef = desiredPose[1,0]
    dxRef = desiredPose[2,0]
    dyRef = desiredPose[3,0]
    vRef = desiredPose[4,0]
    wRef = desiredPose[5,0]

    theta_ref = np.arctan2(dyRef, dxRef)
    qr = np.array([[xRef], [yRef], [theta_ref]])
    return controller.kinematic_control(currentPose, qr, vRef, wRef)

q0 = np.array([[1.1], [0.8], [0]])
tSpan = (0, 50)
dt = 0.01
intg = EulerNumericalIntegrator(dynamic, control, desired, q0, tSpan, dt)
timeSteps, states, desireds, controls = intg.simulation()

plt.plot(states[0,:], states[1,:])
plt.plot(desireds[0,:], desireds[1,:])
# plt.plot(timeSteps, states[0,:])
# plt.plot(timeSteps, states[1,:])
# plt.plot(timeSteps, states[2,:])
plt.show()