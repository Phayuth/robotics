import os
import sys

sys.path.append(str(os.path.abspath(os.getcwd())))

import numpy as np
from matplotlib import animation
import matplotlib.pyplot as plt
from simulator.sim_diffdrive import DiffDrive2DSimulator
from simulator.integrator_euler import EulerNumericalIntegrator
from controllers.differential.pose_basic import DifferentialDrivePoseBasicController

# environment
env = DiffDrive2DSimulator(2)

# controller
controller = DifferentialDrivePoseBasicController(env.robot)


# simulator
def dynamic(currentPose, input):
    return env.robot.forward_external_kinematic(input, currentPose[2, 0])


def desired(currentPose, time):
    return np.array([4.0, 4.0, 0]).reshape(3, 1)


def control(currentPose, desiredPose):
    return controller.kinematic_control(currentPose, desiredPose)


q0 = np.array([1.0, 0.0, -np.pi]).reshape(3, 1)
tSpan = (0, 15)
dt = 0.01
intg = EulerNumericalIntegrator(dynamic, control, desired, q0, tSpan, dt)
timeSteps, states, desireds, controls = intg.simulation()


plt.plot(states[0, :], states[1, :])
# plt.plot(timeSteps, states[0,:])
# plt.plot(timeSteps, states[1,:])
# plt.plot(timeSteps, states[2,:])
plt.show()

env.play_back_path(states, animation)
