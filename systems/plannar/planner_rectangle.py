import os
import sys

wd = os.path.abspath(os.getcwd())
sys.path.append(str(wd))

import numpy as np
np.random.seed(9)

# planner
from planner.sampling_based.rrt_plannerapi import RRTPlannerAPI

# environment
from simulator.sim_rectangle import TaskSpace2DSimulator

# joint value
from datasave.joint_value.experiment_paper import ICRABarnMap

sim = TaskSpace2DSimulator()

plannarConfigSingleTree = {
    "planner": 5,
    "eta": 0.3,
    "subEta": 0.05,
    "maxIteration": 2000,
    "simulator": sim,
    "nearGoalRadius": 0.3,
    "rewireRadius": None,
    "endIterationID": 1,
    "printDebug": True,
    "localOptEnable": True
}


q = ICRABarnMap.PoseSingleDuplicateGoal()
# q = ICRABarnMap.PoseMulti3DuplicateGoal()
xStart = q.xStart
xApp = q.xApp
xGoal = q.xGoal

pm = RRTPlannerAPI.init_normal(xStart, xApp, xGoal, plannarConfigSingleTree)
patha = pm.begin_planner()

pm.plot_2d_config_tree()
pm.plot_performance()