import os
import sys

wd = os.path.abspath(os.getcwd())
sys.path.append(str(wd))

import time
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

# planner
from manipulator_planner import PlannerManipulator

# environment
from copsim.arm_api import UR5eArmCoppeliaSimAPI

# joint value
from datasave.joint_value.pre_record_value import SinglePose, MultiplePoses
from datasave.joint_value.experiment_paper import URHarvesting

# plotter
from planner.rrt_plotter import RRTPlotter


copsimConfigSingleTree = {
    "planner": RRTStar,
    "eta": 0.15,
    "subEta": 0.05,
    "maxIteration": 3000,
    "robotEnvClass": UR5eArmCoppeliaSimAPI,
    "nearGoalRadius": 0.3,
    "rewireRadius": None,
    "endIterationID": 1,
    "print_debug": True,
    "localOptEnable": True
}


copsimConfigDualTree = {
    "planner": RRTStarConnect,
    "eta": 0.15,
    "subEta": 0.05,
    "maxIteration": 3000,
    "robotEnvClass": UR5eArmCoppeliaSimAPI,
    "nearGoalRadius": None,
    "rewireRadius": None,
    "endIterationID": 1,
    "print_debug": True,
    "localOptEnable": True
}


# q = SinglePose.Pose6()
# q = MultiplePoses.Pose6()
q = URHarvesting.PoseSingle1()
xStart = q.xStart
xApp = q.xApp
xGoal = q.xGoal

pa = PlannerManipulator(xStart, xApp, xGoal, copsimConfigDualTree)
path = pa.planning()
time.sleep(3)
pa.planner.robotEnvClass.play_back_path(path)

fig, ax = plt.subplots()
RRTPlotter.plot_performance(pa.planner.perfMatrix, ax)
plt.show()