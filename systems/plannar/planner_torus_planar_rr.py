import os
import sys

wd = os.path.abspath(os.getcwd())
sys.path.append(str(wd))

import numpy as np
np.random.seed(9)
import matplotlib.pyplot as plt

# planner
from planner.planner_torus import PlannerTorusManipulator
from planner.sampling_based.rrt_plotter import RRTPlotter

# environment
from simulator.sim_planar_rr import RobotArm2DSimulator

sim = RobotArm2DSimulator()

plannarConfig = {
    "planner": 5,
    "eta": 0.3,
    "subEta": 0.05,
    "maxIteration": 2000,
    "simulator": sim,
    "nearGoalRadius": None,
    "rewireRadius": None,
    "endIterationID": 1,
    "printDebug": True,
    "localOptEnable": True
}

xStart = np.array([0.0, -3.0]).reshape(2,1)
xApp = np.array([1.470, 0.2]).reshape(2,1)
xGoal = np.array([1.711, 0.2]).reshape(2,1)

pa = PlannerTorusManipulator(xStart, xApp, xGoal, plannarConfig)

patha = pa.planning()
fig, ax = plt.subplots()
fig.set_size_inches(w=3.40067, h=3.40067)
fig.tight_layout()
plt.xlim((-2*np.pi, 2*np.pi))
plt.ylim((-2*np.pi, 2*np.pi))
# RRTPlotter.plot_2d_config_single_tree(pa.planner, patha, ax)
RRTPlotter.plot_2d_config_dual_tree(pa.planner, patha, ax)

plt.show()

fig, ax = plt.subplots()
RRTPlotter.plot_performance(pa.planner.perfMatrix, ax)
plt.show()