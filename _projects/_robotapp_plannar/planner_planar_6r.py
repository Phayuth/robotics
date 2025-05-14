import os
import sys

sys.path.append(str(os.path.abspath(os.getcwd())))

import numpy as np

np.random.seed(9)

from planner.sampling_based.rrt_plannerapi import RRTPlannerAPI
from simulator.sim_planar_6r import RobotArm6RSimulator

sim = RobotArm6RSimulator()

plannarConfigDualTreea = {
    "planner": 5,
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

xStart = np.array([0, 0, 0, 0, 0, 0]).reshape(6,1)
xApp = np.array([np.pi, 0, 0, 0, 0, 0]).reshape(6,1)
xGoal = np.array([np.pi, 0, 0, 0, 0, 0]).reshape(6,1)

pm = RRTPlannerAPI.init_normal(xStart, xApp, xGoal, plannarConfigDualTreea)
patha = pm.begin_planner()

pm.plot_performance()
