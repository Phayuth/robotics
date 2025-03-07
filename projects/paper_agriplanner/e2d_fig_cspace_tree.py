import os
import sys

sys.path.append(str(os.path.abspath(os.getcwd())))

import matplotlib.pyplot as plt
import numpy as np
from planner.sampling_based.rrt_plannerapi import RRTPlannerAPI
from e2d_fig_cspace import plot_cobs_concavehull
from e2d_parameters import sim, xApp, xGoal, xStart, PaperLengths

np.random.seed(9)
plt.rcParams["svg.fonttype"] = "none"

# 16, Birrt
# 12, RRTConnect
# 13, RRTStarConnect

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

# wd = 6.68459 * 0.32
# ht = 6.68459 * 0.32
# fig = plt.figure(figsize=(wd, ht), frameon=True, layout="tight", dpi=600)
# ax = plt.subplot(1, 1, 1)
# ax.set_aspect("equal")
# pm.plot_2d_complete_external(ax)
# # ax.set_xlabel("$\\theta_1$")
# # ax.set_ylabel("$\\theta_2$")
# # ax.set_xticklabels([])
# # ax.set_yticklabels([])
# # ax.set_xticks([])
# # ax.set_yticks([])
# ax.set_xticklabels(["$-\\pi$", "$\\theta_1$", "$\\pi$"])
# ax.set_yticklabels(["$-\\pi$", "$\\theta_2$", "$\\pi$"])
# ax.set_xticks([-np.pi, 0, np.pi])
# ax.set_yticks([-np.pi, 0, np.pi])
# ax.set_xlim(-np.pi, np.pi)
# ax.set_ylim(-np.pi, np.pi)
# fig.tight_layout()

wd = 0.30 * PaperLengths.latex_textwidth_inch
ht = 0.30 * PaperLengths.latex_textwidth_inch
fig = plt.figure(figsize=(wd, ht), frameon=True, layout="tight", dpi=600)
ax = plt.subplot(1, 1, 1)
ax.set_aspect("equal")

plot_cobs_concavehull(ax)
pm.plot_2d_tree_path_state_external(ax)
for axis in ["top", "bottom", "left", "right"]:
    ax.spines[axis].set_linewidth(1)
ax.set_xticklabels(["$-\pi\$", "$\\theta_1\$", "$\pi\$"])
ax.set_yticklabels(["$-\pi\$", "$\\theta_2\$", "$\pi\$"])
ax.set_xticks([-2.8, 0, 2.8])
ax.set_yticks([-2.8, 0, 2.8])
ax.set_xlim(-np.pi, np.pi)
ax.set_ylim(-np.pi, np.pi)
ax.tick_params(axis="both", which="major", direction="in", length=0, width=1, colors="black", grid_color="gray", grid_alpha=0.2, labelsize=0.1)
ax.tick_params(axis="both", which="minor", direction="in", length=0, width=1, colors="black", grid_color="gray", grid_alpha=0.1, labelsize=0.1)


# exp1_birrt
# exp1_birrt_lg
if planconfig["planner"] == 16:
    str_p = "exp1_birrt"
elif planconfig["planner"] == 12:
    str_p = "exp1_rrtcnt"
elif planconfig["planner"] == 13:
    str_p = "exp1_rrtstarcnt"

if planconfig["localOptEnable"]:
    str_p += "_lg"

plt.savefig(f"/home/yuth/{str_p}.svg", bbox_inches="tight", format="svg", transparent=True)
