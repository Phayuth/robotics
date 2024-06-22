import os
import sys

wd = os.path.abspath(os.getcwd())
sys.path.append(str(wd))

import time
import numpy as np
np.random.seed(9)
import matplotlib.pyplot as plt

# planner
from planner.planner_manipulator import PlannerManipulator
from planner.sampling_based.rrt_plotter import RRTPlotter

# environment
from simulator.sim_ur5e_api import UR5eArmCoppeliaSimAPI
from spatial_geometry.utils import Utilities

simu = UR5eArmCoppeliaSimAPI()

copsimConfigDualTree = {
    "planner": 5,
    "eta": 0.15,
    "subEta": 0.05,
    "maxIteration": 500,
    "simulator": simu,
    "nearGoalRadius": None,
    "rewireRadius": None,
    "endIterationID": 1,
    "printDebug": True,
    "localOptEnable": True
}


xStart = np.array([-1.66792136827577, -1.95428576091908, -1.50018072128295, -4.45825996021413, -1.65081054369081, -3.14131814638246]).reshape(6, 1)
xApp = np.array([-0.024881664906637013, -1.8307167492308558, -1.5795478820800781, -2.6565920315184535, -1.555636231099264, -3.134223286305563]).reshape(6,1)
xGoal = np.array([-0.0027387777911584976, -1.9624139271178187, -1.4210033416748047, -2.6216727695860804, -1.4972699324237269, -3.134235207234518]).reshape(6,1)


XAPPALT = Utilities.find_shifted_value(xApp)
XGOALALT = Utilities.find_shifted_value(xGoal)

for i in range(XAPPALT.shape[1]):
    xa = XAPPALT[:,i,np.newaxis]
    xg = XGOALALT[:,i,np.newaxis]

    pa = PlannerManipulator(xStart, xa, xg, copsimConfigDualTree, wrap=False)
    pq = pa.planning()
    time.sleep(3)


    fig, ax = plt.subplots()
    RRTPlotter.plot_performance(pa.planner.perfMatrix, ax)
    plt.show()

    simu.play_back_path(pq)

    input()