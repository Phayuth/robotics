import os
import sys

sys.path.append(str(os.path.abspath(os.getcwd())))

import numpy as np
from simulator.sim_planar_rr import RobotArm2DSimulator
import matplotlib.pyplot as plt
from spatial_geometry.spatial_transformation import RigidBodyTransformation as rbt
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

thetas = np.hstack((q0, q1, q2, q3, q1i, q2i, q3i))

# figure setup
wd = 3.19423 * 0.8
ht = 3.19423 * 0.8
fig = plt.figure(figsize=(wd, ht), frameon=True, layout="tight", dpi=600)
ax = plt.subplot(1, 1, 1)
ax.set_aspect("equal")
ax.set_xlim(-4, 4)
ax.set_ylim(-2, 4)
ax.axhline(color="gray", alpha=0.4)
ax.axvline(color="gray", alpha=0.4)

# ax.grid(False)
# ax.set_xticks([-4, -2, 0, 2, 4])
# ax.set_yticks([-4, -2, 0, 2, 4])
# ax.set_xlabel("$x$", fontsize=11)
# ax.set_ylabel("$y$", fontsize=11)

# remove stuff
ax.grid(False)
ax.set_xticklabels(["$x$"])
ax.set_yticklabels(["$y$"])
ax.set_xticks([0])
ax.set_yticks([0])

sim.plot_taskspace()
robot.plot_arm(thetas, ax, shadow=False, colors=["indigo", "red", "blue", "green", "red", "blue", "green"])
c = plt.Circle((cx, cy), rc, alpha=0.4, edgecolor="g", facecolor=None, fill=False, linestyle="--")
ax.add_patch(c)
ax.plot([cx], [cy], "g*")

plt.savefig("/home/yuth/exp1_wspace.png", bbox_inches="tight", transparent=True)
