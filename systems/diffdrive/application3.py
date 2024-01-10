"""
Application 3
Robot : DiffDrive
Planner : RRT and Smooth with Curve fit
Controller : Trajectory Control + pose basic control at goal region
"""
import os
import sys
wd = os.path.abspath(os.getcwd())
sys.path.append(str(wd))

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
plt.style.use("seaborn")
np.random.seed(1010)
from matplotlib.animation import FuncAnimation
from robot.differential import DifferentialDrive
from controllers.differential.trajectory_backstepping import DifferentialDriveBackSteppingTrajectoryController
from controllers.differential.pose_basic import DifferentialDrivePoseBasicController
from planner.rrtbase import RRTMobileRobotBase
from maps.map import application1_map
from trajectory_planner.traj_polynomial import poly_deg7, trajectory_poly_deg7

# robot and env obs
robot = DifferentialDrive(wheelRadius=0.03, baseLength=0.3, baseWidth=0.3)
obsMap = application1_map()
qStart = np.array([[4], [1], [0.6 * np.pi]])
qGoal = np.array([[-9], [9], [0.6 * np.pi]])
q = qStart.copy()

# pre env plot
plt.plot([-10, 10, 10, -10, -10], [-10, -10, 10, 10, -10], color="red") # border
for obs in obsMap:
    obs.plot()
robot.plot_robot(qStart, plt)
robot.plot_robot(qGoal, plt)
plt.show()

# planner and result path
planner = RRTMobileRobotBase(robot, obsMap, qStart, qGoal, eta=0.3, maxIteration=2000)
planner.planning()
path = planner.search_path()
pathX = [node.x for node in path]
pathY = [node.y for node in path]

# Fit the line equation
tEnd = 60
time = np.linspace(0, tEnd, len(pathX))
sigma = np.ones(len(pathX))
sigma[[0, -1]] = 0.01
poptX, pcovX = curve_fit(poly_deg7, time, pathX, sigma=sigma)
poptY, pcovY = curve_fit(poly_deg7, time, pathY, sigma=sigma)
fig4, axes = plt.subplots(2, 1, sharex='all')
axes[0].plot(time, pathX, 'ro')
axes[0].plot(time, poly_deg7(time, *poptX))
axes[1].plot(time, pathY, 'ro')
axes[1].plot(time, poly_deg7(time, *poptY))
plt.show()

timeSmooth = np.linspace(0, tEnd, 1000)
pathSmoothX = [poly_deg7(t, *poptX) for t in timeSmooth]
pathSmoothY = [poly_deg7(t, *poptY) for t in timeSmooth]
qref = np.array([pathSmoothX, pathSmoothY]).T
# pre env plot
plt.plot([-10, 10, 10, -10, -10], [-10, -10, 10, 10, -10], color="red") # border
for obs in obsMap:
    obs.plot()
plt.plot(qref[:, 0], qref[:, 1])
robot.plot_robot(qStart, plt)
robot.plot_robot(qGoal, plt)
plt.show()


# controller
controllerBackStep = DifferentialDriveBackSteppingTrajectoryController(robot=robot)
controllerPoseFwd = DifferentialDrivePoseBasicController(robot=robot)

# simulation params, save history
t = 0
Ts = 0.03
qHistCur = q.copy()
qHistRef = np.array([0,0,0]).reshape(3,1)
tHist = [0]
controllerSwitch = False
while t < tEnd:
    xRef, xdot, xddot = trajectory_poly_deg7(t, *poptX)
    yRef, ydot, yddot = trajectory_poly_deg7(t, *poptY)
    vr = np.sqrt((xdot**2 + ydot**2))
    wr = ((xdot*yddot - ydot*xddot)) / ((xdot**2 + ydot**2))
    theta_ref = np.arctan2(ydot, xdot)
    qr = np.array([[xRef], [yRef], [theta_ref]])

    # controller
    if controllerSwitch is False:
        if np.linalg.norm([(qGoal[0,0] - q[0,0]),(qGoal[1,0] - q[1,0])]) < 2:
            controllerSwitch = True
        bodyVeloControl = controllerBackStep.kinematic_control(q, qr, vr, wr)

    elif controllerSwitch is True:
        bodyVeloControl = controllerPoseFwd.kinematic_control(q, qGoal)

    # store history
    qHistCur = np.hstack((qHistCur, q))
    qHistRef = np.hstack((qHistRef, qr))

    tHist.append(t)

    # Euler Intergral Update new path
    dq = robot.forward_external_kinematic(bodyVeloControl, q[2, 0])
    q = q + dq*Ts
    t += Ts

# plot
qHistCur = qHistCur.T
fig, ax = plt.subplots()
ax.grid(True)
ax.set_aspect("equal")
ax.set_xlim([-10, 10])
ax.set_ylim([-10, 10])
ax.plot([-10, 10, 10, -10, -10], [-10, -10, 10, 10, -10], color="red") # border
robot.plot_robot(qStart, ax)
robot.plot_robot(qGoal, ax)
ax.plot(qref[:, 0], qref[:, 1])
for obs in obsMap:
    obs.plot()
# link
link1, = ax.plot([], [], 'teal')
link2, = ax.plot([], [], 'olive')
link3, = ax.plot([], [], 'teal')
link4, = ax.plot([], [], 'olive')
link5, = ax.plot([], [], 'olive')
link6, = ax.plot([], [], 'teal')
def update(frame):
    link = robot.robot_link(qHistCur[frame].reshape(3,1))
    link1.set_data([link[0][0], link[2][0]], [link[0][1], link[2][1]])
    link2.set_data([link[1][0], link[2][0]], [link[1][1], link[2][1]])
    link3.set_data([link[2][0], link[3][0]], [link[2][1], link[3][1]])
    link4.set_data([link[3][0], link[4][0]], [link[3][1], link[4][1]])
    link5.set_data([link[4][0], link[1][0]], [link[4][1], link[1][1]])
    link6.set_data([link[0][0], link[3][0]], [link[0][1], link[3][1]])
animation = FuncAnimation(fig, update, frames=(qHistCur.shape[0]), interval=10)
plt.show()

tHist = np.array(tHist)
print(f"==>> tHist: \n{tHist}")

fig4, axes = plt.subplots(2, 1, sharex='all')
axes[0].plot(tHist, qHistCur[:,0], 'r:', label="desired x")
axes[0].plot(tHist, poly_deg7(tHist, *poptX), label="actual x")
axes[1].plot(tHist, qHistCur[:,1], 'g:', label="desired y")
axes[1].plot(tHist, poly_deg7(tHist, *poptY), label="actual y")
fig4.legend()
plt.show()