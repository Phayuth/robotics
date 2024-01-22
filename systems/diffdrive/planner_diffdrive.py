import os
import sys
wd = os.path.abspath(os.getcwd())
sys.path.append(str(wd))

import numpy as np
np.random.seed(0)
import matplotlib.pyplot as plt
from planner.planner_mobilerobot import PlannerMobileRobot
from planner.sampling_based.rrt_plotter import RRTPlotter
from simulator.sim_diffdrive import DiffDrive2DSimulator


sim = DiffDrive2DSimulator()
plannarConfigDualTreea = {
    "planner": 1,
    "eta": 0.3,
    "subEta": 0.05,
    "maxIteration": 6000,
    "simulator": sim,
    "nearGoalRadius": None,
    "rewireRadius": None,
    "endIterationID": 1,
    "printDebug": True,
    "localOptEnable": False
}

xStart = np.array([29.50, 6.10]).reshape(2,1)
xApp = np.array([15.00, 2.80]).reshape(2,1)
xGoal = np.array([15.90, 2.30]).reshape(2,1)

pa = PlannerMobileRobot(xStart, xApp, xGoal, plannarConfigDualTreea)

patha = pa.planning()
fig, ax = plt.subplots()
ax.set_axis_off()
fig.set_size_inches(w=3.40067, h=3.40067)
fig.tight_layout()
# plt.xlim((-np.pi, np.pi))
# plt.ylim((-np.pi, np.pi))
RRTPlotter.plot_2d_config_single_tree(pa.planner, patha, ax)
# RRTPlotter.plot_2d_config_dual_tree(pa.planner, patha, ax)
plt.show()

fig, ax = plt.subplots()
RRTPlotter.plot_performance(pa.planner.perfMatrix, ax)
plt.show()