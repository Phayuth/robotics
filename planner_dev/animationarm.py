import os
import sys

wd = os.path.abspath(os.getcwd())
sys.path.append(str(wd))

import matplotlib.pyplot as plt
import numpy as np
from devplaner import DevPlanner
from matplotlib.animation import FuncAnimation

np.random.seed(9)
from scipy.optimize import curve_fit
from map.taskmap_geo_format import two_near_ee_for_devplanner

from planner_util.extract_path_class import extract_path_class_6d
from planner_util.plot_util import plot_joint_6d
from robot.planar_sixdof import PlanarSixDof
from util.dictionary_pretty import print_dict

robot = PlanarSixDof()

thetaInit = np.array([0, 0, 0, 0, 0, 0]).reshape(6, 1)
thetaGoal = np.array([1.2, 0, -0.2, 0, -1.2, -0.2]).reshape(6, 1)
thetaApp = np.array([1.3, 0, -0.3, 0, -1.21, -0.16]).reshape(6, 1)
obsList = two_near_ee_for_devplanner()

# plot pre planning
fig1, ax1 = plt.subplots()
ax1.set_aspect("equal")
ax1.set_title("Pre Planning Plot")
robot.plot_arm(thetaInit, plt_axis=ax1)
robot.plot_arm(thetaApp, plt_axis=ax1)
robot.plot_arm(thetaGoal, plt_axis=ax1)
for obs in obsList:
    obs.plot()
plt.show()

planner = DevPlanner(robot, obsList, thetaInit, thetaApp, thetaGoal, eta=0.1, maxIteration=5000)
path = planner.planning()
print_dict(planner.perfMatrix)

pathX, pathY, pathZ, pathP, pathQ, pathR = extract_path_class_6d(path)
index = np.arange(0,len(pathX), 1)

# Create a figure and axis for the animation
fig, ax = plt.subplots()

# Set the x and y axis limits
ax.set_xlim(-1, 3.5)
ax.set_ylim(-0.5, 2)
ax.axvline(x=0, c="green")
ax.axhline(y=0, c="green")
# Initialize the line objects for each link of the arm
link1, = ax.plot([], [], 'cyan', linewidth=3)
link2, = ax.plot([], [], 'tan', linewidth=3)
link3, = ax.plot([], [], 'olive', linewidth=3)
link4, = ax.plot([], [], 'navy', linewidth=3)
link5, = ax.plot([], [], 'lime', linewidth=3)
link6, = ax.plot([], [], 'peru', linewidth=3)
endpoint, = ax.plot([], [], 'ro')
for obs in obsList:
    obs.plot()
# Animation update function
def update(frame):
    # Get the current joint angles
    theta = np.array([[pathX[frame]], [pathY[frame]], [pathZ[frame]], [pathP[frame]], [pathQ[frame]], [pathR[frame]]])
    
    # Compute the arm configuration using forward kinematics
    arm_config = robot.forward_kinematic(theta, return_link_pos=True)
    
    # Update the line objects with the new arm configuration
    link1.set_data([arm_config[0][0], arm_config[1][0]], [arm_config[0][1], arm_config[1][1]])
    link2.set_data([arm_config[1][0], arm_config[2][0]], [arm_config[1][1], arm_config[2][1]])
    link3.set_data([arm_config[2][0], arm_config[3][0]], [arm_config[2][1], arm_config[3][1]])
    link4.set_data([arm_config[3][0], arm_config[4][0]], [arm_config[3][1], arm_config[4][1]])
    link5.set_data([arm_config[4][0], arm_config[5][0]], [arm_config[4][1], arm_config[5][1]])
    link6.set_data([arm_config[5][0], arm_config[6][0]], [arm_config[5][1], arm_config[6][1]])
    endpoint.set_data(arm_config[6][0], arm_config[6][1])

animation = FuncAnimation(fig, update, frames=len(index), interval=100, repeat=False)
plt.show()






# Optimization stage, I want to fit the current theta to time and use that information to inform sampling
time = np.linspace(0, 1, len(pathX))

def quintic5deg(x, a, b, c, d, e, f):
    return a * x**5 + b * x**4 + c * x**3 + d * x**2 + e*x*f

# Fit the line equation
poptX, pcovX = curve_fit(quintic5deg, time, pathX)
poptY, pcovY = curve_fit(quintic5deg, time, pathY)
poptZ, pcovZ = curve_fit(quintic5deg, time, pathZ)
poptP, pcovP = curve_fit(quintic5deg, time, pathP)
poptQ, pcovQ = curve_fit(quintic5deg, time, pathQ)
poptR, pcovR = curve_fit(quintic5deg, time, pathR)

timeSmooth = np.linspace(0, 1, 100)
# # plot after planning
# fig3, ax3 = plt.subplots()
# ax3.set_aspect("equal")
# ax3.set_title("After Planning Plot")
# for obs in obsList:
#     obs.plot()
# for i in range(timeSmooth.shape[0]):
#     robot.plot_arm(np.array([quintic5deg(timeSmooth[i], *poptX),
#                                 quintic5deg(timeSmooth[i], *poptY),
#                                 quintic5deg(timeSmooth[i], *poptZ),
#                                 quintic5deg(timeSmooth[i], *poptP),
#                                 quintic5deg(timeSmooth[i], *poptQ),
#                                 quintic5deg(timeSmooth[i], *poptR)]).reshape(6, 1), plt_axis=ax3)
#     plt.pause(0.1)
# plt.show()

# fig4, axes = plt.subplots(6, 1, sharex='all')
# axes[0].plot(time, pathX, 'ro')
# axes[0].plot(time, quintic5deg(time, *poptX))
# axes[1].plot(time, pathY, 'ro')
# axes[1].plot(time, quintic5deg(time, *poptY))
# axes[2].plot(time, pathZ, 'ro')
# axes[2].plot(time, quintic5deg(time, *poptZ))
# axes[3].plot(time, pathP, 'ro')
# axes[3].plot(time, quintic5deg(time, *poptP))
# axes[4].plot(time, pathQ, 'ro')
# axes[4].plot(time, quintic5deg(time, *poptQ))
# axes[5].plot(time, pathR, 'ro')
# axes[5].plot(time, quintic5deg(time, *poptR))
# plt.show()


# Create a figure and axis for the animation
fig, ax = plt.subplots()

# Set the x and y axis limits
ax.set_xlim(-1, 3.5)
ax.set_ylim(-0.5, 2)
ax.axvline(x=0, c="green")
ax.axhline(y=0, c="green")
# Initialize the line objects for each link of the arm
link1, = ax.plot([], [], 'cyan', linewidth=3)
link2, = ax.plot([], [], 'tan', linewidth=3)
link3, = ax.plot([], [], 'olive', linewidth=3)
link4, = ax.plot([], [], 'navy', linewidth=3)
link5, = ax.plot([], [], 'lime', linewidth=3)
link6, = ax.plot([], [], 'peru', linewidth=3)
endpoint, = ax.plot([], [], 'ro')
for obs in obsList:
    obs.plot()

index = np.arange(0,len(timeSmooth), 1)

def update(frame):
    theta = np.array([[quintic5deg(timeSmooth[frame], *poptX)],
                      [quintic5deg(timeSmooth[frame], *poptY)],
                      [quintic5deg(timeSmooth[frame], *poptZ)],
                      [quintic5deg(timeSmooth[frame], *poptP)],
                      [quintic5deg(timeSmooth[frame], *poptQ)],
                      [quintic5deg(timeSmooth[frame], *poptR)]])
    
    # Compute the arm configuration using forward kinematics
    arm_config = robot.forward_kinematic(theta, return_link_pos=True)
    
    # Update the line objects with the new arm configuration
    link1.set_data([arm_config[0][0], arm_config[1][0]], [arm_config[0][1], arm_config[1][1]])
    link2.set_data([arm_config[1][0], arm_config[2][0]], [arm_config[1][1], arm_config[2][1]])
    link3.set_data([arm_config[2][0], arm_config[3][0]], [arm_config[2][1], arm_config[3][1]])
    link4.set_data([arm_config[3][0], arm_config[4][0]], [arm_config[3][1], arm_config[4][1]])
    link5.set_data([arm_config[4][0], arm_config[5][0]], [arm_config[4][1], arm_config[5][1]])
    link6.set_data([arm_config[5][0], arm_config[6][0]], [arm_config[5][1], arm_config[6][1]])
    endpoint.set_data(arm_config[6][0], arm_config[6][1])

animation = FuncAnimation(fig, update, frames=len(index), interval=100, repeat=False)
plt.show()