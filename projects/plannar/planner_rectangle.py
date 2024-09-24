import os
import sys

sys.path.append(str(os.path.abspath(os.getcwd())))

import numpy as np
np.random.seed(9)

from planner.sampling_based.rrt_plannerapi import RRTPlannerAPI
from simulator.sim_rectangle import TaskSpace2DSimulator
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

pm.plot_2d_complete()
pm.plot_performance()