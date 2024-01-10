import os
import sys

wd = os.path.abspath(os.getcwd())
sys.path.append(str(wd))

import numpy as np
from spatial_geometry.utils import Utilities
from planner.sampling_based.rrt_base import RRTBaseMulti
from planner.sampling_based.rrt_connect import RRTConnectMulti
from planner.sampling_based.rrt_star import RRTStarMulti
from planner.sampling_based.rrt_informed import RRTInformedMulti
from planner.sampling_based.rrt_star_connect import RRTStarConnectMulti
from planner.sampling_based.rrt_informed_connect import RRTInformedConnect
from planner.sampling_based.rrt_star_quick import RRTStarQuickMulti
from planner.sampling_based.rrt_star_connect_quick import RRTStarConnectQuickMulti


class PlannerTorusManipulator:
    """
    Planner ID :

    0. RRTBase Multi
    1. RRTStar Multi
    2. RRTInformed Multi
    3. RRTStarQuick Multi
    4. RRTConnect Multi
    5. RRTStarConnect Multi
    6. RRTStarConnectQuick Multi

    """
    def __init__(self, xStart, xApp, xGoal, config):
        # initial configuration
        xAppCandidate = Utilities.find_shifted_value(xApp)
        xGoalCandidate = Utilities.find_shifted_value(xGoal)
        xApp = [xAppCandidate[:,i,np.newaxis] for i in range(xAppCandidate.shape[1])]
        xGoal = [xGoalCandidate[:,i,np.newaxis] for i in range(xGoalCandidate.shape[1])]

        self.planningAlg = [# single tree
                            RRTBaseMulti,             # 0
                            RRTStarMulti,             # 1
                            RRTInformedMulti,         # 2
                            RRTStarQuickMulti,        # 3

                            # dual tree
                            RRTConnectMulti,          # 4
                            RRTStarConnectMulti,      # 5
                            RRTStarConnectQuickMulti] # 6

        self.planner = self.planningAlg[config["planner"]](xStart, xApp, xGoal, config)

    def planning(self):
        return self.planner.begin_planner()