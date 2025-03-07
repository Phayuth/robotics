import os
import sys

sys.path.append(str(os.path.abspath(os.getcwd())))

import matplotlib.pyplot as plt
from e2d_parameters import PaperLengths, thetas, rc, cx, cy, sim, robot, NonMobileTaskMap

plt.rcParams["svg.fonttype"] = "none"

# figure setup
# wd = 3.19423 * 0.8
# ht = 3.19423 * 0.8
wd = 0.30 * PaperLengths.latex_textwidth_inch
ht = 0.30 * PaperLengths.latex_textwidth_inch
fig = plt.figure(figsize=(wd, ht), frameon=True, layout="tight", dpi=600)
ax = plt.subplot(1, 1, 1)
ax.set_aspect("equal")

ax.set_xlim(NonMobileTaskMap.paper_ijcas2025_xlim)
ax.set_ylim(NonMobileTaskMap.paper_ijcas2025_ylim)
ax.axhline(color="gray", alpha=0.4)
ax.axvline(color="gray", alpha=0.4)
ax.grid(False)
ax.set_xticklabels(["$x\$"])
ax.set_yticklabels(["$y\$"])
ax.set_xticks([0])
ax.set_yticks([0])
ax.tick_params(axis="both", which="major", direction="in", length=0, width=1, colors="black", grid_color="gray", grid_alpha=0.2, labelsize=0.1)
ax.tick_params(axis="both", which="minor", direction="in", length=0, width=1, colors="black", grid_color="gray", grid_alpha=0.1, labelsize=0.1)


sim.plot_taskspace()
robot.plot_arm(thetas, ax, shadow=False, colors=["indigo", "red", "blue", "green", "red", "blue", "green"])
c = plt.Circle((cx, cy), rc, alpha=0.4, edgecolor="g", facecolor=None, fill=False, linestyle="--")
ax.add_patch(c)
ax.plot([cx], [cy], "g*")

# plt.savefig("/home/yuth/exp1_wspace.png", bbox_inches="tight", transparent=True)
plt.savefig("/home/yuth/exp1_wspace.svg", bbox_inches="tight", format="svg", transparent=True)
