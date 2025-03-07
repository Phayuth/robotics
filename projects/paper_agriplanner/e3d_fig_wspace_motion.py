import os
import sys

sys.path.append(str(os.path.abspath(os.getcwd())))
import matplotlib.pyplot as plt
import numpy as np
from planner.sampling_based.rrt_plannerapi import RRTPlannerAPI
from e2d_parameters import sim, xApp, xGoal, xStart, PaperLengths, NonMobileTaskMap, robot, cx, cy, rc

np.random.seed(9)


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

pm = RRTPlannerAPI.init_normal(xStart, xApp, xGoal, planconfig)
path = pm.begin_planner()


# figure setup
wd = 3.19423 * 0.8
ht = 3.19423 * 0.8
fig = plt.figure(figsize=(wd, ht), frameon=True, layout="tight", dpi=600)
ax = plt.subplot(1, 1, 1)
ax.set_aspect("equal")
ax.set_xlim(NonMobileTaskMap.paper_ijcas2025_xlim)
ax.set_ylim(NonMobileTaskMap.paper_ijcas2025_ylim)
ax.axhline(color="gray", alpha=0.4)
ax.axvline(color="gray", alpha=0.4)

# remove stuff
ax.grid(False)
ax.set_xticklabels(["$x$"])
ax.set_yticklabels(["$y$"])
ax.set_xticks([0])
ax.set_yticks([0])

sim.plot_taskspace()
robot.plot_arm(path, ax, shadow=True)
c = plt.Circle((cx, cy), rc, alpha=0.4, edgecolor="g", facecolor=None, fill=False, linestyle="--")
ax.add_patch(c)
ax.plot([cx], [cy], "g*")

plt.savefig("/home/yuth/exp3_motion.png", bbox_inches="tight", transparent=True)
