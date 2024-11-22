import os
import sys

sys.path.append(str(os.path.abspath(os.getcwd())))

import numpy as np
from matplotlib import animation
import matplotlib.pyplot as plt
from simulator.sim_diffdrive import DiffDrive2DSimulator
from simulator.integrator_euler import EulerNumericalIntegrator
from controllers.differential.trajectory_backstepping import DifferentialDriveBackSteppingTrajectoryController
from trajectory_generator.traj_primitive import circle


# environment
env = DiffDrive2DSimulator(2)

# controller
controller = DifferentialDriveBackSteppingTrajectoryController(env.robot)


# simulator
def dynamic(currentPose, input):
    return env.robot.forward_external_kinematic(input, currentPose[2, 0])


def desired(currentPose, time):
    xRef, yRef, vr, wr, ydot, xdot, vdotref, wdotref = circle(time)
    return np.array([[xRef], [yRef], [vr], [wr], [xdot], [ydot]])


def control(currentPose, desiredPose):
    xRef = desiredPose[0, 0]
    yRef = desiredPose[1, 0]
    vr = desiredPose[2, 0]
    wr = desiredPose[3, 0]
    xdot = desiredPose[4, 0]
    ydot = desiredPose[5, 0]
    thetaRef = np.arctan2(ydot, xdot)
    qr = np.array([[xRef], [yRef], [thetaRef]])
    return controller.kinematic_control(currentPose, qr, vr, wr)


q0 = np.array([[6], [0], [0]])
tSpan = (0, 50)
dt = 0.01
intg = EulerNumericalIntegrator(dynamic, control, desired, q0, tSpan, dt)
timeSteps, states, desireds, controls = intg.simulation()

plt.plot(states[0, :], states[1, :])
# plt.plot(timeSteps, states[0,:])
# plt.plot(timeSteps, states[1,:])
# plt.plot(timeSteps, states[2,:])
plt.show()

env.play_back_path(states, animation)
