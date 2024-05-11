import os
import sys

wd = os.path.abspath(os.getcwd())
sys.path.append(str(wd))

import numpy as np
np.random.seed(9)
import matplotlib.pyplot as plt

# planner
from planner.planner_manipulator import PlannerManipulator
from planner.sampling_based.rrt_plotter import RRTPlotter

# environment
from simulator.sim_planar_rr import RobotArm2DSimulator

# joint value
from datasave.joint_value.experiment_paper import Experiment2DArm

sim = RobotArm2DSimulator()

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
    "localOptEnable": False
}


q = Experiment2DArm.PoseSingle()
# q = Experiment2DArm.PoseMulti()
xStart = q.xStart
xApp = q.xApp
xGoal = q.xGoal

pm = PlannerManipulator(xStart, xApp, xGoal, plannarConfigDualTreea)

patha = pm.planning()
fig, ax = plt.subplots()
ax.set_axis_off()
fig.set_size_inches(w=3.40067, h=3.40067)
fig.tight_layout()
plt.xlim((-np.pi, np.pi))
plt.ylim((-np.pi, np.pi))
# RRTPlotter.plot_2d_config_single_tree(pa.planner, patha, ax)
RRTPlotter.plot_2d_config_dual_tree(pm.planner, patha, ax)

plt.show()

fig, ax = plt.subplots()
RRTPlotter.plot_performance(pm.planner.perfMatrix, ax)
plt.show()