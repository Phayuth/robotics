"""
Application 1
Robot : DiffDrive
Planner : RRT
Controller : Path Segment
"""
import os
import sys
wd = os.path.abspath(os.getcwd())
sys.path.append(str(wd))

import numpy as np
import matplotlib.pyplot as plt
np.random.seed(1010)
from matplotlib.animation import FuncAnimation
from controllers.differential.path_segment import DifferentialDrivePathSegmentTrackingController
from planner.planner_mobilerobot import PlannerMobileRobot
from simulator.sim_diffdrive import DiffDrive2DSimulator
from planner.sampling_based.rrt_star_connect import RRTStarConnect

# robot and env obs
qStart = np.array([[4], [1]])
qAux = np.array([[-8.9], [8.9]])
qGoal = np.array([[-9], [9]])

# planner and result path
plannarConfigDualTreea = {
    "planner": RRTStarConnect,
    "eta": 0.3,
    "subEta": 0.05,
    "maxIteration": 500,
    "simulator": DiffDrive2DSimulator,
    "nearGoalRadius": None,
    "rewireRadius": None,
    "endIterationID": 1,
    "printDebug": True,
    "localOptEnable": True
}

planner = PlannerMobileRobot(qStart, qAux, qGoal, plannarConfigDualTreea)
path = planner.planning()
print(f"==>> path: {path}")
fig, ax = plt.subplots()
ax.set_axis_off()

p = [p.config for p in path]
qref = np.concatenate(p, axis=1).T
print(f"==>> qref.shape: {qref.shape}")

# controller
env = DiffDrive2DSimulator()
controller = DifferentialDrivePathSegmentTrackingController(robot=env.robot, referenceSegment=qref)

# simulation params, save history
t = 0
Ts = 0.03
q = np.array([[4], [1], [0.6 * np.pi]])
qHistCur = np.array([[4], [1], [0.6 * np.pi]])
tHist = [0]

while t < 800:
    # controller
    bodyVeloControl = controller.kinematic_control(q)

    # store history
    qHistCur = np.hstack((qHistCur, q))
    tHist.append(t * Ts)

    # Euler Intergral Update new path
    dq = env.robot.forward_external_kinematic(bodyVeloControl, q[2, 0])
    q = q + dq*Ts
    t += 1

# plot
qHistCur = qHistCur.T
fig, ax = plt.subplots()
ax.grid(True)
ax.set_aspect("equal")
ax.set_xlim([-10, 10])
ax.set_ylim([-10, 10])
ax.plot([-10, 10, 10, -10, -10], [-10, -10, 10, 10, -10], color="red") # border
# env.robot.plot_robot(qStart, ax)
# env.robot.plot_robot(qGoal, ax)
ax.plot(qref[:, 0], qref[:, 1])
for obs in env.taskMapObs:
    obs.plot()
# link
link1, = ax.plot([], [], 'teal')
link2, = ax.plot([], [], 'olive')
link3, = ax.plot([], [], 'teal')
link4, = ax.plot([], [], 'olive')
link5, = ax.plot([], [], 'olive')
link6, = ax.plot([], [], 'teal')
def update(frame):
    link = env.robot.robot_link(qHistCur[frame].reshape(3,1))
    link1.set_data([link[0][0], link[2][0]], [link[0][1], link[2][1]])
    link2.set_data([link[1][0], link[2][0]], [link[1][1], link[2][1]])
    link3.set_data([link[2][0], link[3][0]], [link[2][1], link[3][1]])
    link4.set_data([link[3][0], link[4][0]], [link[3][1], link[4][1]])
    link5.set_data([link[4][0], link[1][0]], [link[4][1], link[1][1]])
    link6.set_data([link[0][0], link[3][0]], [link[0][1], link[3][1]])
animation = FuncAnimation(fig, update, frames=(qHistCur.shape[0]), interval=10)
plt.show()