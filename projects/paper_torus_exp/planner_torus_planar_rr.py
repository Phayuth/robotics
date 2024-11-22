import os
import sys

sys.path.append(str(os.path.abspath(os.getcwd())))

import numpy as np
np.random.seed(9)

from planner.sampling_based.rrt_plannerapi import RRTPlannerAPI
from simulator.sim_planar_rr import RobotArm2DSimulator

sim = RobotArm2DSimulator(torusspace=True)

plannarConfig = {
    "planner": 9,
    "eta": 0.3,
    "subEta": 0.05,
    "maxIteration": 10000,
    "simulator": sim,
    "nearGoalRadius": None,
    "rewireRadius": None,
    "endIterationID": 1,
    "printDebug": True,
    "localOptEnable": False,
}

# xStart = np.array([0.0, 0.0]).reshape(2,1)
# # xApp = np.array([4.6, -5.77]).reshape(2,1)
# xApp = np.array([3.0, 0.5]).reshape(2,1)

xStart = np.array([-2, 2.5]).reshape(2, 1)
xApp = np.array([3, -2]).reshape(2, 1)

xGoal = xApp.copy()

pa = RRTPlannerAPI.init_alt_q_torus(xStart, xApp, xGoal, plannarConfig, None)
patha = pa.begin_planner()

pa.plot_2d_complete()
pa.plot_performance()
# plt.savefig("/home/yuth/exp1_wspace.png", bbox_inches="tight", transparent=True)


from matplotlib import animation
sim.play_back_path(patha, animation)