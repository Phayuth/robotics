import os
import sys

wd = os.path.abspath(os.getcwd())
sys.path.append(str(wd))

import numpy as np
np.random.seed(0)
import matplotlib.pyplot as plt

# planner
from planner.planner_mobilerobot import PlannerMobileRobot
from planner.sampling_based.rrt_plotter import RRTPlotter

# environment
from simulator.sim_diffdrive import DiffDrive2DSimulator

from trajectory_generator.traj_spline_interpolation import CubicSpline2D


plannarConfigDualTreea = {
    "planner": 7,
    "eta": 0.3,
    "subEta": 0.05,
    "maxIteration": 3000,
    "simulator": DiffDrive2DSimulator,
    "nearGoalRadius": None,
    "rewireRadius": None,
    "endIterationID": 1,
    "printDebug": True,
    "localOptEnable": True
}


xStart = np.array([0, 0]).reshape(2,1)
xApp = np.array([8.4, -4.1]).reshape(2,1)
xGoal = np.array([8.5, -4.2]).reshape(2,1)

pa = PlannerMobileRobot(xStart, xApp, xGoal, plannarConfigDualTreea)

patha = pa.planning()
fig, ax = plt.subplots()
ax.set_axis_off()
fig.set_size_inches(w=3.40067, h=3.40067)
fig.tight_layout()
# plt.xlim((-np.pi, np.pi))
# plt.ylim((-np.pi, np.pi))
# RRTPlotter.plot_2d_config_single_tree(pa.planner, patha, ax)
RRTPlotter.plot_2d_config_dual_tree(pa.planner, patha, ax)

plt.show()

fig, ax = plt.subplots()
RRTPlotter.plot_performance(pa.planner.perfMatrix, ax)
plt.show()


# pathInterpolate = np.array([pathX, pathY]).T
# sp = CubicSpline2D(pathInterpolate)
# rx, ry, ryaw, rk, s = sp.course(ds=0.01)
# plt.grid(True)
# plt.axvline(x=0, c="yellow")
# plt.axhline(y=0, c="yellow")
# planner.plot_env(after_plan=True)
# plt.plot(rx, ry, "*r", label="Cubic spline path")
# plt.plot([node.x for node in path], [node.y for node in path], color='blue', label="Discretize path")
# plt.legend()
# plt.show()