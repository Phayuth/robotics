import os
import sys

sys.path.append(str(os.path.abspath(os.getcwd())))

import numpy as np
from simulator.sim_planar_rr import RobotArm2DSimulator
import matplotlib.pyplot as plt
from matplotlib import ticker

env = RobotArm2DSimulator(torusspace=False)

wd = 6.68459 * 0.32
ht = 6.68459 * 0.32
fig = plt.figure(figsize=(wd, ht), frameon=True, layout="tight", dpi=600)
ax = plt.subplot(1, 1, 1)
ax.set_aspect("equal")
env.plot_cspace(ax)
# ax.set_xlabel("$\\theta_1$")
# ax.set_ylabel("$\\theta_2$")
ax.set_xticklabels(["$-\\pi$", "$\\theta_1$", "$\\pi$"])
ax.set_yticklabels(["$-\\pi$", "$\\theta_2$","$\\pi$"])
ax.set_xticks([-np.pi, 0, np.pi])
ax.set_yticks([-np.pi, 0, np.pi])
ax.set_xlim(-np.pi, np.pi)
ax.set_ylim(-np.pi, np.pi)
# ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
# ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
# ax.tick_params(axis="both", which="major", direction="in", length=5, width=1, colors="black", grid_color="gray", grid_alpha=0.2)
# ax.tick_params(axis="both", which="minor", direction="in", length=3, width=1, colors="black", grid_color="gray", grid_alpha=0.1)
plt.savefig("/home/yuth/exp1_cspace.png", bbox_inches="tight")