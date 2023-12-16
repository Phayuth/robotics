import os
import sys

wd = os.path.abspath(os.getcwd())
sys.path.append(str(wd))

import numpy as np
np.random.seed(0)
import matplotlib.pyplot as plt

# planning algorithm
from planner.rrt_base import RRTBase, RRTBaseMulti
from planner.rrt_connect import RRTConnect, RRTConnectMulti
from planner.rrt_star import RRTStar, RRTStarMulti
from planner.rrt_informed import RRTInformed, RRTInformedMulti
from planner.rrt_star_connect import RRTStarConnect, RRTStarConnectMulti
from planner.rrt_informed_connect import RRTInformedConnect
from planner.rrt_connect_ast_informed import RRTConnectAstInformed
from planner.rrt_star_quick import RRTStarQuick, RRTStarQuickMulti
from planner.rrt_star_connect_quick import RRTStarConnectQuick, RRTStarConnectQuickMulti

# planner
from manipulator_planner import PlannerManipulator

# environment
from map.sim_2d_env import RobotArm2DEnvironment, TaskSpace2DEnvironment

# joint value
from datasave.joint_value.experiment_paper import Experiment2DArm, ICRABarnMap

# plotter
from planner.rrt_plotter import RRTPlotter


plannarConfigSingleTree = {
    "planner": RRTStarQuick,
    "eta": 0.3,
    "subEta": 0.05,
    "maxIteration": 2000,
    "robotEnvClass": TaskSpace2DEnvironment,
    "nearGoalRadius": 0.3,
    "rewireRadius": None,
    "endIterationID": 1,
    "print_debug": True,
    "localOptEnable": True
}


plannarConfigDualTreea = {
    "planner": RRTStarConnect,
    "eta": 0.3,
    "subEta": 0.05,
    "maxIteration": 2000,
    "robotEnvClass": TaskSpace2DEnvironment,
    "nearGoalRadius": None,
    "rewireRadius": None,
    "endIterationID": 4,
    "print_debug": True,
    "localOptEnable": True
}

# plannarConfigDualTreeb = {
#     "planner": RRTStarConnect,
#     "eta": 0.3,
#     "subEta": 0.05,
#     "maxIteration": 2000,
#     "robotEnvClass": TaskSpace2DEnvironment,
#     "nearGoalRadius": None,
#     "rewireRadius": None,
#     "endIterationID": 1,
#     "print_debug": True,
#     "localOptEnable": True
# }

q = ICRABarnMap.PoseSingle()
# q = ICRABarnMap.PoseMulti3()
xStart = q.xStart
xApp = q.xApp
xGoal = q.xGoal

pa = PlannerManipulator(xStart, xApp, xGoal, plannarConfigDualTreea)
# pb = PlannerManipulator(xStart, xApp, xGoal, plannarConfigDualTreeb)

patha = pa.planning()
# pathb = pb.planning()
fig, ax = plt.subplots()
ax.set_axis_off()
fig.set_size_inches(w=3.40067, h=3.40067)
fig.tight_layout()
plt.xlim((-np.pi, np.pi))
plt.ylim((-np.pi, np.pi))
# RRTPlotter.plot_2d_config_single_tree(pa.planner, patha, ax)
# RRTPlotter.plot_2d_path(pathb, ax)
RRTPlotter.plot_2d_config_dual_tree(pa.planner, patha, ax)

plt.show()

fig, ax = plt.subplots()
RRTPlotter.plot_performance(pa.planner.perfMatrix, ax)
plt.show()