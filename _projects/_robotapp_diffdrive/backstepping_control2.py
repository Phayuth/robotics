"""
Robot : DiffDrive
Controller : Trajectory Control with switch to pose baisc control near goal region
"""

import os
import sys

sys.path.append(str(os.path.abspath(os.getcwd())))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import animation
from controllers.differential.trajectory_backstepping import DifferentialDriveBackSteppingTrajectoryController
from controllers.differential.pose_basic import DifferentialDrivePoseBasicController
from simulator.sim_diffdrive import DiffDrive2DSimulator
from simulator.integrator_euler import EulerNumericalIntegrator
from datasave.joint_value.pre_record_value import PreRecordedPathMobileRobot
from trajectory_generator.traj_interpolator import BSplineSmoothingUnivariant

# path
path = np.array(PreRecordedPathMobileRobot.warehouse_path)
print(f"> path.shape: {path.shape}")
xc = path[:, 0]
yc = path[:, 1]
tc = np.linspace(0, 100, len(xc))

# fitted trajectory : 1 trajectory for all point
# polyCx = np.polynomial.polynomial.polyfit(tc, xc, 30)
# polyCy = np.polynomial.polynomial.polyfit(tc, yc, 100)
# polyX = np.polynomial.Polynomial(polyCx)
# polyY = np.polynomial.Polynomial(polyCy)
# xm = polyX(tc)
# ym = polyY(tc)

# fitted treajectory : smoothing spline
cs = BSplineSmoothingUnivariant(tc, path.T, smoothc=5)
xym = cs.eval_pose(tc)
xm = xym[0]
ym = xym[1]

# plot
fig = plt.figure(figsize=(8, 8))
gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1])
ax1 = plt.subplot(gs[:, 0])
ax1.plot(xc, yc, "ro")
ax1.plot(xm, ym)
ax1.set_title("xy")
ax2 = plt.subplot(gs[0, 1])
ax2.plot(tc, xc, "ro")
ax2.plot(tc, xm)
ax2.set_title("x")
ax3 = plt.subplot(gs[1, 1])
ax3.plot(tc, yc, "ro")
ax3.plot(tc, ym)
ax3.set_title("y")
plt.tight_layout()
plt.show()

# create robot and controller
env = DiffDrive2DSimulator()
controller = DifferentialDriveBackSteppingTrajectoryController(robot=env.robot)
controllerPoseFwd = DifferentialDrivePoseBasicController(robot=env.robot)


# simulator
def dynamic(currentPose, input):
    return env.robot.forward_external_kinematic(input, currentPose[2, 0])


# # fitted trajectory : Spline trajectory
def desired(currentPose, time):
    xym = cs.eval_pose(time)
    xRef = xym[0]
    yRef = xym[1]

    xydm = cs.eval_velo(time)
    xRefDot = xydm[0]
    yRefDot = xydm[1]

    xyddm = cs.eval_accel(time)
    xRefDDot = xyddm[0]
    yRefDDot = xyddm[1]

    vr = np.sqrt((xRefDot**2 + yRefDot**2))
    wr = ((xRefDot * yRefDDot - yRefDot * xRefDDot)) / ((xRefDot**2 + yRefDot**2))

    return np.array([[xRef], [yRef], [vr], [wr], [xRefDot], [yRefDot]])


# fitted trajectory : 1 trajectory for all point
# def desired(currentPose, time):
#     xRef = polyX(time)
#     yRef = polyY(time)

#     xRefDot = polyX.deriv()(time)
#     yRefDot = polyY.deriv()(time)

#     xRefDDot = polyX.deriv().deriv()(time)
#     yRefDDot = polyY.deriv().deriv()(time)

#     vr = np.sqrt((xRefDot**2 + yRefDot**2))
#     print(f"> vr: {vr}")
#     wr = ((xRefDot * yRefDDot - yRefDot * xRefDDot)) / ((xRefDot**2 + yRefDot**2))

#     return np.array([[xRef], [yRef], [vr], [wr], [xRefDot], [yRefDot]])


def control(currentPose, desiredPose):
    if np.linalg.norm([(currentPose[0, 0] - qg[0, 0]), (currentPose[1, 0] - qg[1, 0])]) > 1.5:
        xRef = desiredPose[0, 0]
        yRef = desiredPose[1, 0]
        vr = desiredPose[2, 0]
        wr = desiredPose[3, 0]
        xdot = desiredPose[4, 0]
        ydot = desiredPose[5, 0]
        thetaRef = np.arctan2(ydot, xdot)
        qr = np.array([[xRef], [yRef], [thetaRef]])
        return controller.kinematic_control(currentPose, qr, vr, wr)
    else:
        return controllerPoseFwd.kinematic_control(currentPose, qg)


q0 = np.array([[15], [2.5], [np.pi / 2]])
qg = np.array([[12], [13.5]])
tSpan = (0, 100)
dt = 0.01
intg = EulerNumericalIntegrator(dynamic, control, desired, q0, tSpan, dt)
timeSteps, states, desireds, controls = intg.simulation()

env.play_back_path(states, animation)
