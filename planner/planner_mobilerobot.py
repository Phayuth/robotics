import os
import sys

wd = os.path.abspath(os.getcwd())
sys.path.append(str(wd))

from planner.sampling_based.rrt_base import RRTBase
from planner.sampling_based.rrt_connect import RRTConnect
from planner.sampling_based.rrt_star import RRTStar
from planner.sampling_based.rrt_informed import RRTInformed
from planner.sampling_based.rrt_star_connect import RRTStarConnect
from planner.sampling_based.rrt_informed_connect import RRTInformedConnect
from planner.sampling_based.rrt_star_quick import RRTStarQuick
from planner.sampling_based.rrt_star_connect_quick import RRTStarConnectQuick


class PlannerMobileRobot:
    """
    Planner ID :

    0. RRTBase
    1. RRTStar
    2. RRTInformed
    3. RRTStarQuick
    4. RRTConnect
    5. RRTStarConnect
    6. RRTInformedConnect
    7. RRTStarConnectQuick

    """
    def __init__(self, xStart, xApp, xGoal, config):
        # process pose, interest in 2D x,y only. rotation is neglected
        xStart = xStart[0:2]
        xApp = xApp[0:2]
        xGoal = xGoal[0:2]

        self.planningAlg = [# single tree
                            RRTBase,                 # 0
                            RRTStar,                 # 1
                            RRTInformed,             # 2
                            RRTStarQuick,            # 3

                            # dual tree
                            RRTConnect,              # 4
                            RRTStarConnect,          # 5
                            RRTInformedConnect,      # 6
                            RRTStarConnectQuick,     # 7
                            ]

        self.planner = self.planningAlg[config["planner"]](xStart, xApp, xGoal, config)

    def planning(self):
        return self.planner.begin_planner()