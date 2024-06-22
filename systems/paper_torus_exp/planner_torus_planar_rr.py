import os
import sys

wd = os.path.abspath(os.getcwd())
sys.path.append(str(wd))

import numpy as np
np.random.seed(9)

# planner
from planner.sampling_based.rrt_plannerapi import RRTPlannerAPI

# environment
from simulator.sim_planar_rr import RobotArm2DSimulator

sim = RobotArm2DSimulator(torusspace=True)

plannarConfig = {
    "planner": 9,
    "eta": 0.3,
    "subEta": 0.05,
    "maxIteration": 2000,
    "simulator": sim,
    "nearGoalRadius": None,
    "rewireRadius": None,
    "endIterationID": 1,
    "printDebug": True,
    "localOptEnable": False
}

xStart = np.array([0.0, 0.0]).reshape(2,1)
xApp = np.array([4.6, -5.77]).reshape(2,1)
xGoal = np.array([4.6, -5.77]).reshape(2,1)

pa = RRTPlannerAPI.init_alt_q_torus(xStart, xApp, xGoal, plannarConfig, None)
patha = pa.begin_planner()

pa.plot_2d_config_tree()
pa.plot_performance()