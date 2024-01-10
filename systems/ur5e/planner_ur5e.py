import os
import sys

wd = os.path.abspath(os.getcwd())
sys.path.append(str(wd))

import time
import numpy as np
np.random.seed(0)
import matplotlib.pyplot as plt

# planner
from planner.planner_manipulator import PlannerManipulator
from planner.sampling_based.rrt_plotter import RRTPlotter

# environment
from simulator.sim_ur5e_api import UR5eArmCoppeliaSimAPI

# joint value
from datasave.joint_value.pre_record_value import SinglePose, MultiplePoses
from datasave.joint_value.experiment_paper import URHarvesting


copsimConfigDualTree = {
    "planner": 5,
    "eta": 0.15,
    "subEta": 0.05,
    "maxIteration": 3000,
    "simulator": UR5eArmCoppeliaSimAPI,
    "nearGoalRadius": None,
    "rewireRadius": None,
    "endIterationID": 1,
    "printDebug": True,
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
pa.planner.simulator.play_back_path(path)

fig, ax = plt.subplots()
RRTPlotter.plot_performance(pa.planner.perfMatrix, ax)
plt.show()