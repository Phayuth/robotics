"""
Application 4
Robot : DiffDrive
Planner : Path
Controller : Pure Pursuit

"""
import os
import sys
wd = os.path.abspath(os.getcwd())
sys.path.append(str(wd))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from robot.differential import DifferentialDrive
from trajectory_planner.path_pre_segment import warehouse_path as path1
from controllers.differential.path_purepursuit import DifferentialDrivePurePursuitController
from maps.map import warehouse_map

robot = DifferentialDrive(wheelRadius=0.03, baseLength=0.3, baseWidth=0.3)
controller = DifferentialDrivePurePursuitController(path1, loop_mode=True)
q = np.array([15.06048387044, 2.496073852304, 0]).reshape(3, 1)

# simulation params, save history
t = 0
Ts = 0.03
qHistCur = q.copy()
tHist = [0]

while t < 10000:
    # controller
    bodyVeloControl = controller.kinematic_control(q)

    # store history
    qHistCur = np.hstack((qHistCur, q))
    tHist.append(t * Ts)

    # Euler Intergral Update new path
    dq = robot.forward_external_kinematic(bodyVeloControl, q[2, 0])
    q = q + dq * Ts
    q[2, 0] = controller.angle_normalize(q[2, 0])
    t += 1

# plot
qHistCur = qHistCur.T
fig, ax = plt.subplots()
ax.grid(True)
ax.set_aspect("equal")
ax.set_xlim([0, 30])
ax.set_ylim([0, 18])
pathRef = np.array(path1)
ax.plot(pathRef[:, 0], pathRef[:, 1], '--', color='grey')
obsList = warehouse_map()
for obs in obsList:
    obs.plot()
plt.text(25, 2.5, "Delivery Vehicle", bbox=dict(facecolor='yellow', alpha=0.5))
plt.text(15, 2.5, "Charging Station", bbox=dict(facecolor='yellow', alpha=0.5))
plt.text(1, 1, "Control Office", bbox=dict(facecolor='yellow', alpha=0.5))
plt.text(25, 14, "Head Office", bbox=dict(facecolor='yellow', alpha=0.5))
plt.text(2, 10, "Large\nStorage", bbox=dict(facecolor='yellow', alpha=0.5))
plt.text(6, 10, "Large\nStorage", bbox=dict(facecolor='yellow', alpha=0.5))
plt.text(13, 16.5, "Small\nStorage", bbox=dict(facecolor='yellow', alpha=0.5))
# link
link1, = ax.plot([], [], 'teal')
link2, = ax.plot([], [], 'olive')
link3, = ax.plot([], [], 'teal')
link4, = ax.plot([], [], 'olive')
link5, = ax.plot([], [], 'olive')
link6, = ax.plot([], [], 'teal')

def update(frame):
    link = robot.robot_link(qHistCur[frame].reshape(3, 1))
    link1.set_data([link[0][0], link[2][0]], [link[0][1], link[2][1]])
    link2.set_data([link[1][0], link[2][0]], [link[1][1], link[2][1]])
    link3.set_data([link[2][0], link[3][0]], [link[2][1], link[3][1]])
    link4.set_data([link[3][0], link[4][0]], [link[3][1], link[4][1]])
    link5.set_data([link[4][0], link[1][0]], [link[4][1], link[1][1]])
    link6.set_data([link[0][0], link[3][0]], [link[0][1], link[3][1]])

animation = animation.FuncAnimation(fig, update, frames=(qHistCur.shape[0]), interval=1)
plt.show()

# using_rotation = False
# numOfFrames = 400

# # animation
# fig = plt.figure()
# trajectory_lines = plt.plot([], '-', color='orange', linewidth=4)
# trajectory_line = trajectory_lines[0]
# heading_lines = plt.plot([], '-', color='red')
# heading_line = heading_lines[0]
# connection_lines = plt.plot([], '-', color='green')
# connection_line = connection_lines[0]
# poses = plt.plot([], 'o', color='black', markersize=10)
# pose = poses[0]
# pathForGraph = np.array(path1)
# plt.plot(pathForGraph[:, 0], pathForGraph[:, 1], '--', color='grey')

# plt.axis("scaled")
# plt.xlim(-6, 6)
# plt.ylim(-4, 4)
# dt = 50
# xs = [currentPos[0, 0]]
# ys = [currentPos[1, 0]]

# def pure_pursuit_animation(frame):
#     # call pure_pursuit_step to get info
#     goalPt, turnVel = controller.kinematic_control(currentPos)

#     # model: 200rpm drive with 18" width
#     #               rpm   /s  circ   feet
#     maxLinVelfeet = 200 / 60 * np.pi * 4 / 12
#     #               rpm   /s  center angle   deg
#     maxTurnVelDeg = 200 / 60 * np.pi * 4 / 9 * 180 / np.pi

#     # update x and y, but x and y stays constant here
#     stepDis = controller.linearVel / 100 * maxLinVelfeet * dt / 1000
#     currentPos[0, 0] += stepDis * np.cos(currentPos[2, 0] * np.pi / 180)
#     currentPos[1, 0] += stepDis * np.sin(currentPos[2, 0] * np.pi / 180)
#     heading_line.set_data([currentPos[0, 0], currentPos[0, 0] + 0.5 * np.cos(currentPos[2, 0] / 180 * np.pi)], [currentPos[1, 0], currentPos[1, 0] + 0.5 * np.sin(currentPos[2, 0] / 180 * np.pi)])
#     connection_line.set_data([currentPos[0, 0], goalPt[0]], [currentPos[1, 0], goalPt[1]])
#     currentPos[2, 0] += turnVel / 100 * maxTurnVelDeg * dt / 1000

#     if using_rotation == False:
#         currentPos[2, 0] = currentPos[2, 0] % 360
#         if currentPos[2, 0] < 0:
#             currentPos[2, 0] += 360

#     # rest of the animation code
#     xs.append(currentPos[0, 0])
#     ys.append(currentPos[1, 0])

#     pose.set_data((currentPos[0, 0], currentPos[1, 0]))
#     trajectory_line.set_data(xs, ys)

# anim = animation.FuncAnimation(fig, pure_pursuit_animation, frames=numOfFrames, interval=50)
# plt.show()
