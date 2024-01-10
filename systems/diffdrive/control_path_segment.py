import os
import sys
wd = os.path.abspath(os.getcwd())
sys.path.append(str(wd))

import numpy as np
import matplotlib.pyplot as plt
from robot.mobile.differential import DifferentialDrive
from controllers.differential.path_segment import DifferentialDrivePathSegmentTrackingController
from simulator.integrator_euler import EulerNumericalIntegrator

# create robot and controller
qref = np.array([[3, 0], [6, 4], [3, 4], [3, 1], [0, 3], [0, 5], [-1, 7], [-5, 0], [-6, 3]])
robot = DifferentialDrive(wheelRadius=0.03, baseLength=0.3, baseWidth=0.3)
controller = DifferentialDrivePathSegmentTrackingController(robot=robot, referenceSegment=qref)

# simulator
def dynamic(currentPose, input):
    return robot.forward_external_kinematic(input, currentPose[2,0])

def desired(currentPose, time):
    return np.array([0.0, 0.0, 0]).reshape(3, 1) # isn't needed in this type of control mode

def control(currentPose, desiredPose):
    return controller.kinematic_control(currentPose)

q0 = np.array([[5], [1], [0.6 * np.pi]])
tSpan = (0, 32)
dt = 0.01
intg = EulerNumericalIntegrator(dynamic, control, desired, q0, tSpan, dt)
timeSteps, states, desireds, controls = intg.simulation()

plt.grid(True)
plt.plot(qref[:, 0], qref[:, 1])
plt.plot(states[0,:], states[1,:])
# plt.plot(timeSteps, states[0,:])
# plt.plot(timeSteps, states[1,:])
# plt.plot(timeSteps, states[2,:])
plt.show()