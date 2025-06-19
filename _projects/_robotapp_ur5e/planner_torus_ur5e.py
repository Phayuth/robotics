import os
import sys

sys.path.append(str(os.path.abspath(os.getcwd())))

import time
import numpy as np

np.random.seed(9)

from planner.sampling_based.rrt_plannerapi import RRTPlannerAPI
from simulator.sim_ur5e_api import UR5eArmCoppeliaSimAPI
from spatial_geometry.utils import Utils

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


xStart = np.array([-1.667, -1.954, -1.500, -4.458, -1.650, -3.141]).reshape(6, 1)
xApp = np.array([-0.024, -1.830, -1.579, -2.656, -1.555, -3.134]).reshape(6, 1)
xGoal = np.array([-0.002, -1.962, -1.421, -2.621, -1.497, -3.134]).reshape(6, 1)


XAPPALT = Utils.find_shifted_value(xApp)
XGOALALT = Utils.find_shifted_value(xGoal)

for i in range(XAPPALT.shape[1]):
    xa = XAPPALT[:, i, np.newaxis]
    xg = XGOALALT[:, i, np.newaxis]

    pa = RRTPlannerAPI.init_normal(xStart, xa, xg, copsimConfigDualTree)
    pq = pa.begin_planner()
    time.sleep(3)

    pa.plot_performance()
    simu.play_back_path(pq)

    input()
