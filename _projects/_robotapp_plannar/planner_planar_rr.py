import os
import sys

sys.path.append(str(os.path.abspath(os.getcwd())))

import numpy as np

np.random.seed(9)

from planner.sampling_based.rrt_plannerapi import RRTPlannerAPI
from simulator.sim_planar_rr import RobotArm2DSimulator
from datasave.joint_value.experiment_paper import Experiment2DArm

sim = RobotArm2DSimulator()

plannarConfigDualTreea = {
    "planner": 4,
    "eta": 0.3,
    "subEta": 0.05,
    "maxIteration": 2000,
    "simulator": sim,
    "nearGoalRadius": None,
    "rewireRadius": None,
    "endIterationID": 1,
    "printDebug": True,
    "localOptEnable": False,
}


# q = Experiment2DArm.PoseSingle()
# # q = Experiment2DArm.PoseMulti()
# xStart = q.xStart
# xApp = q.xApp
# xGoal = q.xGoal

xStart = np.array([0.0, 2.8]).reshape(2, 1)
xApp = np.array([-1.2, -2.3]).reshape(2, 1)
xGoal = np.array([-1.2, -2.3]).reshape(2, 1)


pm = RRTPlannerAPI.init_normal(xStart, xApp, xGoal, plannarConfigDualTreea)
patha = pm.begin_planner()

pm.plot_2d_complete()
pm.plot_performance()


import matplotlib.pyplot as plt

time = np.linspace(0, 1, num=patha.shape[1])
print(f"> time.shape: {time.shape}")
print(patha.shape)

fig, axs = plt.subplots(patha.shape[0], 1, figsize=(10, 15), sharex=True)
for i in range(patha.shape[0]):
    axs[i].plot(time, patha[i, :], color="blue", marker="o", linestyle="dashed", linewidth=2, markersize=12, label=f"Joint Pos {i+1}")
    axs[i].set_ylabel(f"JPos {i+1}")
    axs[i].set_xlim(time[0], time[-1])
    axs[i].legend(loc="upper right")
    axs[i].grid(True)
axs[-1].set_xlabel("Time")
fig.suptitle("Joint Position")
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()
