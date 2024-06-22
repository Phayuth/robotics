import os
import sys
wd = os.path.abspath(os.getcwd())
sys.path.append(str(wd))

import numpy as np
np.random.seed(0)
import matplotlib.pyplot as plt
from planner.sampling_based.rrt_plotter import RRTPlotter
from simulator.sim_diffdrive import DiffDrive2DSimulator
from planner.sampling_based.rrt_plannerapi import RRTPlannerAPI

xStart = np.array([29.50, 6.10]).reshape(2,1)
xApp = np.array([15.00, 2.80]).reshape(2,1)
xGoal = np.array([15.90, 2.30]).reshape(2,1)

sim = DiffDrive2DSimulator()

plannarConfigDualTreea = {
    "planner": 1,
    "eta": 0.3,
    "subEta": 0.05,
    "maxIteration": 3000,
    "simulator": sim,
    "nearGoalRadius": None,
    "rewireRadius": None,
    "endIterationID": 1,
    "printDebug": True,
    "localOptEnable": False
}

pa = RRTPlannerAPI.init_k_element_q(xStart, xApp, xGoal, plannarConfigDualTreea, 2)
patha = pa.begin_planner()

pa.plot_2d_config_tree()
pa.plot_performance()