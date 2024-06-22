import os
import sys

wd = os.path.abspath(os.getcwd())
sys.path.append(str(wd))

import time
import numpy as np

np.random.seed(9)
import matplotlib.pyplot as plt

# planner
from planner.sampling_based.rrt_plannerapi import RRTPlannerAPI

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
    "localOptEnable": True,
}


xStart = np.array([-1.66792136827577, -1.95428576091908, -1.50018072128295, -4.45825996021413, -1.65081054369081, -3.14131814638246]).reshape(6, 1)
xApp = np.array([-0.024881664906637013, -1.8307167492308558, -1.5795478820800781, -2.6565920315184535, -1.555636231099264, -3.134223286305563]).reshape(6, 1)
xGoal = np.array([-0.0027387777911584976, -1.9624139271178187, -1.4210033416748047, -2.6216727695860804, -1.4972699324237269, -3.134235207234518]).reshape(6, 1)

XAPPALT = Utilities.find_alt_config(xApp, np.array(simu.configLimit), None, False)
XGOALALT = Utilities.find_alt_config(xGoal, np.array(simu.configLimit), None, False)

distorg = np.linalg.norm(xApp - xStart)
print(f"> distorg: {distorg}")
distalt = np.linalg.norm((XAPPALT - xStart), axis=0)
print(f"> distalt.shape: {distalt.shape}")
print(f"> distalt: {distalt}")


# i = 0 # 5.157047081477856
# i = 1 # 9.59335527471311
# pa = RRTPlannerAPI.init_normal(xStart, xApp, xGoal, copsimConfigDualTree)
# pq = pa.begin_planner()
# pa.plot_performance()
# simu.play_back_path(pq)


# aaa = np.array([0.0] * 6).reshape(6, 1)
# for i in range(XAPPALT.shape[1]):
#     xa = XAPPALT[:, i, np.newaxis]
#     xg = XGOALALT[:, i, np.newaxis]

#     simu.set_joint_position(simu.jointDynamicHandles, aaa)
#     time.sleep(2)
#     simu.set_joint_position(simu.jointDynamicHandles, xa)
#     print(i, xa)
#     time.sleep(3)


for i in range(XAPPALT.shape[1]):
    xa = XAPPALT[:, i, np.newaxis]
    xg = XGOALALT[:, i, np.newaxis]
    print(i, xa)

    pa = RRTPlannerAPI.init_normal(xStart, xa, xg, copsimConfigDualTree)
    pq = pa.begin_planner()
    time.sleep(3)

    # pa.plot_performance()
    simu.play_back_path(pq)
    input("enter to continue")
