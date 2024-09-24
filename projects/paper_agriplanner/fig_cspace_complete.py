import os
import sys

sys.path.append(str(os.path.abspath(os.getcwd())))
import numpy as np

import matplotlib.pyplot as plt
import numpy as np

np.random.seed(9)
from spatial_geometry.spatial_transformation import RigidBodyTransformation as rbt
from simulator.sim_planar_rr import RobotArm2DSimulator
from planner.sampling_based.rrt_plannerapi import RRTPlannerAPI
from spatial_geometry.utils import Utils


rc = 0.8
cx = -3.0
cy = 0.5
g1 = rbt.conv_polar_to_cartesian(rc, 0.0, cx, cy)
g2 = rbt.conv_polar_to_cartesian(rc, 0.5, cx, cy)  # intended collision
g3 = rbt.conv_polar_to_cartesian(rc, -0.5, cx, cy)

sim = RobotArm2DSimulator()

robot = sim.robot
q0 = np.array([0, 0]).reshape(2, 1)

q1 = robot.inverse_kinematic_geometry(np.array(g1).reshape(2, 1), elbow_option=1)
q2 = robot.inverse_kinematic_geometry(np.array(g2).reshape(2, 1), elbow_option=1)
q3 = robot.inverse_kinematic_geometry(np.array(g3).reshape(2, 1), elbow_option=1)

q1i = robot.inverse_kinematic_geometry(np.array(g1).reshape(2, 1), elbow_option=0)
q2i = robot.inverse_kinematic_geometry(np.array(g2).reshape(2, 1), elbow_option=0)
q3i = robot.inverse_kinematic_geometry(np.array(g3).reshape(2, 1), elbow_option=0)

q1i = Utils.wrap_to_pi(q1i)
q2i = Utils.wrap_to_pi(q2i)
q3i = Utils.wrap_to_pi(q3i)


planconfig = {
    "planner": 13,
    "eta": 0.3,
    "subEta": 0.05,
    "maxIteration": 2000,
    "simulator": sim,
    "nearGoalRadius": None,
    "rewireRadius": None,
    "endIterationID": 1,
    "printDebug": True,
    "localOptEnable": True,
}
# 16, 12, 13
xStart = q0
xApp = [q1, q3, q1i, q2i, q3i]
xGoal = [q1, q3, q1i, q2i, q3i]

pm = RRTPlannerAPI.init_normal(xStart, xApp, xGoal, planconfig)
path = pm.begin_planner()

wd = 6.68459 * 0.32
ht = 6.68459 * 0.32
fig = plt.figure(figsize=(wd, ht), frameon=True, layout="tight", dpi=600)
ax = plt.subplot(1, 1, 1)
ax.set_aspect("equal")
pm.plot_2d_complete_external(ax)
# ax.set_xlabel("$\\theta_1$")
# ax.set_ylabel("$\\theta_2$")
# ax.set_xticklabels([])
# ax.set_yticklabels([])
# ax.set_xticks([])
# ax.set_yticks([])
ax.set_xticklabels(["$-\\pi$", "$\\theta_1$", "$\\pi$"])
ax.set_yticklabels(["$-\\pi$", "$\\theta_2$", "$\\pi$"])
ax.set_xticks([-np.pi, 0, np.pi])
ax.set_yticks([-np.pi, 0, np.pi])
ax.set_xlim(-np.pi, np.pi)
ax.set_ylim(-np.pi, np.pi)
fig.tight_layout()
plt.savefig("/home/yuth/exp1_cspacecomplete.png", bbox_inches="tight", transparent=True)
