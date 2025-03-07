import os
import sys

sys.path.append(str(os.path.abspath(os.getcwd())))

import numpy as np
import matplotlib.pyplot as plt

from spatial_geometry.utils import Utils

from planner.sampling_based.rrt_component import RRTPlotter

from planner.sampling_based.rrt_base import RRTBase, RRTBaseMulti

from planner.sampling_based.rrt_connect import RRTConnect, RRTConnectMulti
from planner.sampling_based.rrt_connect_noneagressive import RRTConnectNoneAgressive, RRTConnectNoneAgressiveMulti

from planner.sampling_based.rrt_star import RRTStar, RRTStarMulti
from planner.sampling_based.rrt_star_quick import RRTStarQuick, RRTStarQuickMulti

from planner.sampling_based.rrt_star_connect import RRTStarConnect, RRTStarConnectMulti
from planner.sampling_based.rrt_star_connect_quick import RRTStarConnectQuick, RRTStarConnectQuickMulti
from planner.sampling_based.rrt_star_connect_noneagressive import RRTStarConnectNoneAgressive, RRTStarConnectNoneAgressiveMulti
from planner.sampling_based.rrt_star_connect_quick_noneagressive import RRTStarConnectQuickNoneAgressive, RRTStarConnectQuickNoneAgressiveMulti

from planner.sampling_based.rrt_informed import RRTInformed, RRTInformedMulti
from planner.sampling_based.rrt_informed_quick import RRTInformedQuick, RRTInformedQuickMulti

from planner.sampling_based.rrt_informed_connect import RRTInformedConnect
from planner.sampling_based.rrt_informed_connect_quick import RRTInformedConnectQuick
from planner.sampling_based.rrt_informed_connect_quick_noneagressive import RRTInformedConnectQuickNoneAgressive


class RRTPlannerAPI:
    """
    Planner ID:
    - RRTBase,  # 0
    - RRTStar,  # 1
    - RRTInformed,  # 2
    - RRTStarQuick,  # 3
    - RRTConnect,  # 4
    - RRTStarConnect,  # 5
    - RRTInformedConnect,  # 6
    - RRTStarConnectQuick,  # 7
    - RRTBaseMulti,  # 8
    - RRTStarMulti,  # 9
    - RRTInformedMulti,  # 10
    - RRTStarQuickMulti,  # 11
    - RRTConnectMulti,  # 12
    - RRTStarConnectMulti,  # 13
    - RRTStarConnectQuickMulti,  # 14
    - RRTConnectNoneAgressive,  # 15
    - RRTConnectNoneAgressiveMulti,  # 16
    ---
    - RRTStarConnectNoneAgressive,  # 17
    - RRTStarConnectNoneAgressiveMulti,  # 18
    - RRTStarConnectQuickNoneAgressive,  # 19
    - RRTStarConnectQuickNoneAgressiveMulti,  # 20
    - RRTInformedQuick,  # 21
    - RRTInformedQuickMulti,  # 22
    - RRTInformedConnectQuick,  # 23
    - RRTInformedConnectQuickNoneAgressive,  # 24
    """

    def __init__(self, xStart, xApp, xGoal, cfg):
        self.cfg = cfg

        self.xStart = xStart
        self.xApp = xApp
        self.xGoal = xGoal

        self.planningAlg = [
            RRTBase,  # 0
            RRTStar,  # 1
            RRTInformed,  # 2
            RRTStarQuick,  # 3
            RRTConnect,  # 4
            RRTStarConnect,  # 5
            RRTInformedConnect,  # 6
            RRTStarConnectQuick,  # 7
            RRTBaseMulti,  # 8
            RRTStarMulti,  # 9
            RRTInformedMulti,  # 10
            RRTStarQuickMulti,  # 11
            RRTConnectMulti,  # 12
            RRTStarConnectMulti,  # 13
            RRTStarConnectQuickMulti,  # 14
            RRTConnectNoneAgressive,  # 15
            RRTConnectNoneAgressiveMulti,  # 16 ---
            RRTStarConnectNoneAgressive,  # 17
            RRTStarConnectNoneAgressiveMulti,  # 18
            RRTStarConnectQuickNoneAgressive,  # 19
            RRTStarConnectQuickNoneAgressiveMulti,  # 20
            RRTInformedQuick,  # 21
            RRTInformedQuickMulti,  # 22
            RRTInformedConnectQuick,  # 23
            RRTInformedConnectQuickNoneAgressive,  # 24
        ]

        self.planner = self.planningAlg[self.cfg["planner"]](xStart, xApp, xGoal, self.cfg)
        self.resultpath = None

    @classmethod
    def init_normal(cls, xStart, xApp, xGoal, cfg):
        return cls(xStart, xApp, xGoal, cfg)

    @classmethod
    def init_warp_q_to_pi(cls, xStart, xApp, xGoal, cfg):
        xStart = Utils.wrap_to_pi(xStart)
        if isinstance(xApp, list):
            xApp = [Utils.wrap_to_pi(x) for x in xApp]
            xGoal = [Utils.wrap_to_pi(x) for x in xGoal]
        else:
            xApp = Utils.wrap_to_pi(xApp)
            xGoal = Utils.wrap_to_pi(xGoal)
        return cls(xStart, xApp, xGoal, cfg)

    @classmethod
    def init_k_element_q(cls, xStart, xApp, xGoal, cfg, k=2):
        # process pose, interest in 2D x,y only. rotation is neglected
        xStart = xStart[0:k]
        xApp = xApp[0:k]
        xGoal = xGoal[0:k]
        return cls(xStart, xApp, xGoal, cfg)

    @classmethod
    def init_alt_q_torus(cls, xStart, xApp, xGoal, cfg, configConstrict):
        configLimit = np.array(cfg["simulator"].configLimit)
        xAppCandidate = Utils.find_alt_config(xApp, configLimit, configConstrict)
        xGoalCandidate = Utils.find_alt_config(xGoal, configLimit, configConstrict)
        xApp = [xAppCandidate[:, i, np.newaxis] for i in range(xAppCandidate.shape[1])]
        xGoal = [xGoalCandidate[:, i, np.newaxis] for i in range(xGoalCandidate.shape[1])]
        return cls(xStart, xApp, xGoal, cfg)

    def begin_planner(self):
        self.resultpath = self.planner.begin_planner()
        return self.resultpath

    def plot_2d_cspace_state(self):
        fig, ax = plt.subplots()
        s = 2.5
        fig.set_size_inches(w=s * 3.40067, h=s * 3.40067)
        fig.tight_layout()
        plt.xlim((self.cfg["simulator"].configLimit[0][0], self.cfg["simulator"].configLimit[0][1]))
        plt.ylim((self.cfg["simulator"].configLimit[1][0], self.cfg["simulator"].configLimit[1][1]))
        RRTPlotter.plot_2d_cspace_state(self.planner, ax)
        plt.show()

    def plot_2d_cspace_state_external(self, ax):
        RRTPlotter.plot_2d_cspace_state(self.planner, ax)

    def plot_2d_tree_path_state(self):
        fig, ax = plt.subplots()
        s = 2.5
        fig.set_size_inches(w=s * 3.40067, h=s * 3.40067)
        fig.tight_layout()
        plt.xlim((self.cfg["simulator"].configLimit[0][0], self.cfg["simulator"].configLimit[0][1]))
        plt.ylim((self.cfg["simulator"].configLimit[1][0], self.cfg["simulator"].configLimit[1][1]))
        RRTPlotter.plot_2d_tree_path_state(self.planner, self.resultpath, ax)
        plt.show()

    def plot_2d_tree_path_state_external(self, ax):
        RRTPlotter.plot_2d_tree_path_state(self.planner, self.resultpath, ax)

    def plot_2d_complete(self):
        fig, ax = plt.subplots()
        s = 2.5
        fig.set_size_inches(w=s * 3.40067, h=s * 3.40067)
        fig.tight_layout()
        plt.xlim((self.cfg["simulator"].configLimit[0][0], self.cfg["simulator"].configLimit[0][1]))
        plt.ylim((self.cfg["simulator"].configLimit[1][0], self.cfg["simulator"].configLimit[1][1]))
        RRTPlotter.plot_2d_complete(self.planner, self.resultpath, ax)
        plt.show()

    def plot_2d_complete_external(self, ax):
        RRTPlotter.plot_2d_complete(self.planner, self.resultpath, ax)

    def plot_performance(self):
        fig, ax = plt.subplots()
        RRTPlotter.plot_performance(self.planner.perfMatrix, ax)
        plt.show()
