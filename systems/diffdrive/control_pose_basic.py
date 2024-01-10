import os
import sys
wd = os.path.abspath(os.getcwd())
sys.path.append(str(wd))

import numpy as np
import matplotlib.pyplot as plt
from robot.mobile.differential import DifferentialDrive
from controllers.differential.pose_basic import DifferentialDrivePoseBasicController
from simulator.integrator_euler import EulerNumericalIntegrator

# create robot and controller
robot = DifferentialDrive(wheelRadius=0, baseLength=0.3, baseWidth=0.3)
controller = DifferentialDrivePoseBasicController(robot=robot)

# simulator
def dynamic(currentPose, input):
    return robot.forward_external_kinematic(input, currentPose[2,0])

def desired(currentPose, time):
    return np.array([4.0, 4.0, 0]).reshape(3, 1)

def control(currentPose, desiredPose):
    return controller.kinematic_control(currentPose, desiredPose)

q0 = np.array([1.0, 0.0, -np.pi]).reshape(3, 1)
tSpan = (0, 15)
dt = 0.01
intg = EulerNumericalIntegrator(dynamic, control, desired, q0, tSpan, dt)
timeSteps, states, desireds, controls = intg.simulation()


plt.plot(states[0,:], states[1,:])
plt.plot(timeSteps, states[0,:])
plt.plot(timeSteps, states[1,:])
plt.plot(timeSteps, states[2,:])
plt.grid(True)
plt.show()

plt.plot(timeSteps, desireds[0,:])
plt.plot(timeSteps, desireds[1,:])
plt.plot(timeSteps, desireds[2,:])
plt.grid(True)
plt.show()