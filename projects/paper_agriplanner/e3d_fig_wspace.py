import os
import sys

sys.path.append(str(os.path.abspath(os.getcwd())))

import matplotlib.pyplot as plt
from e3d_parameters import sim, robot, thetas, cx, cy, rc, NonMobileTaskMap

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
robot.plot_arm(thetas, ax, shadow=False, colors=["indigo", "red", "blue", "green", "red", "blue", "green"])
c = plt.Circle((cx, cy), rc, alpha=0.4, edgecolor="g", facecolor=None, fill=False, linestyle="--")
ax.add_patch(c)
ax.plot([cx], [cy], "g*")
# plt.show()
plt.savefig("/home/yuth/exp3_wspace.png", bbox_inches="tight", transparent=True)
