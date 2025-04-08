import os
import sys

sys.path.append(str(os.path.abspath(os.getcwd())))

import matplotlib.pyplot as plt
from simulator.sim_planar_rr import RobotArm2DSimulator
import numpy as np
from task_map import PaperICCAS2024, PaperTorusIFAC2025

# cspace
torus = False
env = RobotArm2DSimulator(PaperICCAS2024(), torusspace=True)
fig, ax = plt.subplots(1, 1)
colp = env.plot_cspace(ax)
if torus:
    ax.set_xlim([-2 * np.pi, 2 * np.pi])
    ax.set_ylim([-2 * np.pi, 2 * np.pi])
    ax.axhline(y=np.pi, color="gray", alpha=0.4)
    ax.axhline(y=-np.pi, color="gray", alpha=0.4)
    ax.axvline(x=np.pi, color="gray", alpha=0.4)
    ax.axvline(x=-np.pi, color="gray", alpha=0.4)
else:
    ax.set_xlim([-np.pi, np.pi])
    ax.set_ylim([-np.pi, np.pi])
plt.show()

if torus:
    np.save("./datasave/planner_ijcas_data/collisionpoint_so2s.npy", colp)
else:
    np.save("./datasave/planner_ijcas_data/collisionpoint_exts.npy", colp)
