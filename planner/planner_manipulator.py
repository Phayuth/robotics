import os
import sys

wd = os.path.abspath(os.getcwd())
sys.path.append(str(wd))

from spatial_geometry.utils import Utilities
from planner.sampling_based.rrt_base import RRTBase, RRTBaseMulti
from planner.sampling_based.rrt_connect import RRTConnect, RRTConnectMulti
from planner.sampling_based.rrt_connect_noneagressive import RRTConnectNoneAgressive, RRTConnectNoneAgressiveMulti
from planner.sampling_based.rrt_star import RRTStar, RRTStarMulti
from planner.sampling_based.rrt_informed import RRTInformed, RRTInformedMulti
from planner.sampling_based.rrt_star_connect import RRTStarConnect, RRTStarConnectMulti
from planner.sampling_based.rrt_informed_connect import RRTInformedConnect
from planner.sampling_based.rrt_star_quick import RRTStarQuick, RRTStarQuickMulti
from planner.sampling_based.rrt_star_connect_quick import RRTStarConnectQuick, RRTStarConnectQuickMulti


class PlannerManipulator:
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

    8. RRTBaseMulti
    9. RRTStarMulti
    10. RRTInformedMulti
    11. RRTStarQuickMulti

    12. RRTConnectMulti
    13. RRTStarConnectMulti
    14. RRTStarConnectQuickMulti

    15. RRTConnectNoneAgressive - baseline
    16. RRTConnectNoneAgressiveMulti - baseline

    """
    def __init__(self, xStart, xApp, xGoal, config, wrap=True):
        # process joint
        if wrap:
            xStart = Utilities.wrap_to_pi(xStart)
            if isinstance(xApp, list):
                xApp = [Utilities.wrap_to_pi(x) for x in xApp]
                xGoal = [Utilities.wrap_to_pi(x) for x in xGoal]
            else:
                xApp = Utilities.wrap_to_pi(xApp)
                xGoal = Utilities.wrap_to_pi(xGoal)

        self.planningAlg = [ # single tree, single goal
                            RRTBase,                 # 0
                            RRTStar,                 # 1
                            RRTInformed,             # 2
                            RRTStarQuick,            # 3

                            # dual tree, single goal
                            RRTConnect,              # 4
                            RRTStarConnect,          # 5
                            RRTInformedConnect,      # 6
                            RRTStarConnectQuick,     # 7

                            # single tree, multi goal
                            RRTBaseMulti,            # 8
                            RRTStarMulti,            # 9
                            RRTInformedMulti,        # 10
                            RRTStarQuickMulti,       # 11

                            # dual tree, multi goal
                            RRTConnectMulti,         # 12
                            RRTStarConnectMulti,     # 13
                            RRTStarConnectQuickMulti,# 14

                            # baseline
                            RRTConnectNoneAgressive,      # 15
                            RRTConnectNoneAgressiveMulti  # 16
                            ]

        self.planner = self.planningAlg[config["planner"]](xStart, xApp, xGoal, config)

    def planning(self):
        return self.planner.begin_planner()