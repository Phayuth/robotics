import os
import sys

sys.path.append(str(os.path.abspath(os.getcwd())))

import numpy as np
np.random.seed(9)

from planner.sampling_based.rrt_plannerapi import RRTPlannerAPI
from spatial_geometry.utils import Utils
from simulator.sim_planar_rr import RobotArm2DSimulator

sim = RobotArm2DSimulator(torusspace=False)

plannarConfigDualTreea = {
    "planner": 1,
    "eta": 0.3,
    "subEta": 0.05,
    "maxIteration": 1000,
    "simulator": sim,
    "nearGoalRadius": None,
    "rewireRadius": None,
    "endIterationID": 1,
    "printDebug": True,
    "localOptEnable": False,
}

# limt2 = np.array([[-2 * np.pi, 2 * np.pi], [-2 * np.pi, 2 * np.pi]])
# gg = Utils.find_alt_config(xApp, limt2)

# xStart = np.array([0.0, 0.0]).reshape(2,1)
# xApp = np.array([3.0, 0.5]).reshape(2,1)
# xApp = np.array([4.6, -5.77]).reshape(2,1)
# xApp = np.array([4.6, 0.51318531]).reshape(2,1)
# xApp = np.array([-1.68318531, -5.77]).reshape(2,1)
# xApp = np.array([-1.68318531, 0.51318531]).reshape(2,1)

xStart = np.array([-2, 2.5]).reshape(2, 1)
xApp = np.array([3, -2]).reshape(2, 1)

xGoal = xApp.copy()

pm = RRTPlannerAPI.init_normal(xStart, xApp, xGoal, plannarConfigDualTreea)
patha = pm.begin_planner()

pm.plot_2d_complete()
pm.plot_performance()

# thetas = np.hstack((q1, q2, q3)).T
# ax = sim.plot_view(thetas)
# ax = sim.plot_view(patha)
# plt.show()

from matplotlib import animation
sim.play_back_path(patha, animation)
