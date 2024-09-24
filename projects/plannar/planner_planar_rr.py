import os
import sys

sys.path.append(str(os.path.abspath(os.getcwd())))

import numpy as np
np.random.seed(9)

from planner.sampling_based.rrt_plannerapi import RRTPlannerAPI
from simulator.sim_planar_rr import RobotArm2DSimulator
from datasave.joint_value.experiment_paper import Experiment2DArm

sim = RobotArm2DSimulator()

plannarConfigDualTreea = {
    "planner": 4,
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

pm = RRTPlannerAPI.init_normal(xStart, xApp, xGoal, plannarConfigDualTreea)
patha = pm.begin_planner()

pm.plot_2d_complete()
pm.plot_performance()