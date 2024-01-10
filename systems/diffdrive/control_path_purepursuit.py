import os
import sys
wd = os.path.abspath(os.getcwd())
sys.path.append(str(wd))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from robot.mobile.differential import DifferentialDrive
from controllers.differential.path_purepursuit import DifferentialDrivePurePursuitController
from datasave.joint_value.pre_record_value import PreRecordedPathMobileRobot
from simulator.integrator_euler import EulerNumericalIntegrator

# robot and controller
path1 = PreRecordedPathMobileRobot.fig_of_8
robot = DifferentialDrive(wheelRadius=0.03, baseLength=0.3, baseWidth=0.3)
controller = DifferentialDrivePurePursuitController(path1, loopMode=True)

# simulator
def dynamic(currentPose, input):
    return robot.forward_external_kinematic(input, currentPose[2,0])

def desired(currentPose, time):
    return np.array([0.0, 0.0, 0]).reshape(3, 1)

def control(currentPose, desiredPose):
    return controller.kinematic_control(currentPose)

q0 = np.array([0.0, 0.0, 0.0]).reshape(3, 1)
tSpan = (0, 90)
dt = 0.01
intg = EulerNumericalIntegrator(dynamic, control, desired, q0, tSpan, dt)
timeSteps, states, desireds, controls = intg.simulation()

plt.plot(states[0,:], states[1,:])
# plt.plot(timeSteps, states[0,:])
# plt.plot(timeSteps, states[1,:])
# plt.plot(timeSteps, states[2,:])
plt.show()

# # plot
fig, ax = plt.subplots()
ax.grid(True)
ax.set_aspect("equal")
ax.set_xlim([-10, 10])
ax.set_ylim([-10, 10])
ax.plot([-10, 10, 10, -10, -10], [-10, -10, 10, 10, -10], color="red")  # border
ax.plot(states[0,:], states[1,:], '--', color='grey')
# link
link1, = ax.plot([], [], 'teal')
link2, = ax.plot([], [], 'olive')
link3, = ax.plot([], [], 'teal')
link4, = ax.plot([], [], 'olive')
link5, = ax.plot([], [], 'olive')
link6, = ax.plot([], [], 'teal')

def update(frame):
    link = robot.robot_link(states[:,frame].reshape(3, 1))
    link1.set_data([link[0][0], link[2][0]], [link[0][1], link[2][1]])
    link2.set_data([link[1][0], link[2][0]], [link[1][1], link[2][1]])
    link3.set_data([link[2][0], link[3][0]], [link[2][1], link[3][1]])
    link4.set_data([link[3][0], link[4][0]], [link[3][1], link[4][1]])
    link5.set_data([link[4][0], link[1][0]], [link[4][1], link[1][1]])
    link6.set_data([link[0][0], link[3][0]], [link[0][1], link[3][1]])

animation = animation.FuncAnimation(fig, update, frames=(states.shape[1]), interval=10)
plt.show()