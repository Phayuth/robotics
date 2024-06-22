import os
import sys

wd = os.path.abspath(os.getcwd())
sys.path.append(str(wd))

import numpy as np
np.random.seed(9)
import matplotlib.pyplot as plt

# planner
from planner.sampling_based.rrt_plannerapi import RRTPlannerAPI
from spatial_geometry.utils import Utilities

# environment
from simulator.sim_planar_rr import RobotArm2DSimulator

sim = RobotArm2DSimulator(torusspace=True)

plannarConfigDualTreea = {
    "planner": 1,
    "eta": 0.3,
    "subEta": 0.05,
    "maxIteration": 4000,
    "simulator": sim,
    "nearGoalRadius": None,
    "rewireRadius": None,
    "endIterationID": 1,
    "printDebug": True,
    "localOptEnable": False
}

# limt2 = np.array([[-2 * np.pi, 2 * np.pi], [-2 * np.pi, 2 * np.pi]])
# gg = Utilities.find_alt_config(xApp, limt2)

xStart = np.array([0.0, 0.0]).reshape(2,1)

xApp = np.array([4.6, -5.77]).reshape(2,1)
xGoal = np.array([4.6, -5.77]).reshape(2,1)

# xApp = np.array([4.6, 0.51318531]).reshape(2,1)
# xGoal = np.array([4.6, 0.51318531]).reshape(2,1)

# xApp = np.array([-1.68318531, -5.77]).reshape(2,1)
# xGoal = np.array([-1.68318531, -5.77]).reshape(2,1)

# xApp = np.array([-1.68318531, 0.51318531]).reshape(2,1)
# xGoal = np.array([-1.68318531, 0.51318531]).reshape(2,1)

pm = RRTPlannerAPI.init_normal(xStart, xApp, xGoal, plannarConfigDualTreea)
patha = pm.begin_planner()

pm.plot_2d_config_tree()
pm.plot_performance()

# thetas = np.hstack((q1, q2, q3)).T
# ax = sim.plot_view(thetas)
ax = sim.plot_view(patha.T)
plt.show()