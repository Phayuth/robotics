import os
import sys
wd = os.path.abspath(os.getcwd())
sys.path.append(str(wd))

import numpy as np
import matplotlib.pyplot as plt
from robot.mobile.ackerman import AckermanSteer
from controllers.ackerman.trajectory_basic import AckermanTrajectoryBasicController
from simulator.integrator_euler import EulerNumericalIntegrator
from trajectory_generator.traj_primitive import fig_of_8

# create robot and controller
robot = AckermanSteer(wheelRadius=0, baseLength=0.1)
controller = AckermanTrajectoryBasicController(robot)

# simulator
def dynamic(currentPose, input):
    return robot.forward_external_kinematic(input, currentPose[2,0])

def desired(currentPose, time):
    xRef, yRef, _, _, _, _, _, _ = fig_of_8(time)
    phiRef = np.arctan2((yRef - currentPose[1, 0]), (xRef - currentPose[0, 0]))
    return np.array([[xRef], [yRef], [phiRef]])

def control(currentPose, desiredPose):
    return robot.physical_limit(controller.kinematic_control(currentPose, desiredPose))

q0 = np.array([[1.1], [0.8], [0]])
tSpan = (0, 50)
dt = 0.01
intg = EulerNumericalIntegrator(dynamic, control, desired, q0, tSpan, dt)
timeSteps, states, desireds, controls = intg.simulation()

plt.plot(states[0,:], states[1,:])
plt.show()