import os
import sys

wd = os.path.abspath(os.getcwd())
sys.path.append(str(wd))

import time
import numpy as np
from planner.rrt_base import RRTBaseMulti
from planner.rrt_connect import RRTConnectMulti
from planner.rrt_star import RRTStarMulti
from planner.rrt_informed import RRTInformedMulti
from planner.rrt_star_connect import RRTStarConnectMulti


class RRTBaseMulti2D(RRTBaseMulti):

    def __init__(self,
                 xStart,
                 xAppList,
                 xGoalList,
                 eta=0.3,
                 subEta=0.05,
                 maxIteration=2000,
                 numDoF=2,
                 envChoice="Planar",
                 nearGoalRadius=0.3,
                 rewireRadius=None,
                 endIterationID=1,
                 print_debug=False):
        super().__init__(xStart=xStart,
                         xAppList=xAppList,
                         xGoalList=xGoalList,
                         eta=eta,
                         subEta=subEta,
                         maxIteration=maxIteration,
                         numDoF=numDoF,
                         envChoice=envChoice,
                         nearGoalRadius=nearGoalRadius,
                         rewireRadius=rewireRadius,
                         endIterationID=endIterationID,
                         print_debug=print_debug)

    def planning(self):
        timePlanningStart = time.perf_counter_ns()
        self.start()
        path = self.get_path()
        timePlanningEnd = time.perf_counter_ns()
        self.update_perf(timePlanningStart, timePlanningEnd)
        return path


class RRTConnectMulti2D(RRTConnectMulti):

    def __init__(self,
                 xStart,
                 xAppList,
                 xGoalList,
                 eta=0.3,
                 subEta=0.05,
                 maxIteration=2000,
                 numDoF=2,
                 envChoice="Planar",
                 nearGoalRadius=None,
                 rewireRadius=None,
                 endIterationID=1,
                 print_debug=False,
                 localOptEnable=False):
        super().__init__(xStart=xStart,
                         xAppList=xAppList,
                         xGoalList=xGoalList,
                         eta=eta,
                         subEta=subEta,
                         maxIteration=maxIteration,
                         numDoF=numDoF,
                         envChoice=envChoice,
                         nearGoalRadius=nearGoalRadius,
                         rewireRadius=rewireRadius,
                         endIterationID=endIterationID,
                         print_debug=print_debug, 
                         localOptEnable=localOptEnable)

    def planning(self):
        timePlanningStart = time.perf_counter_ns()
        self.start()
        path = self.get_path()
        timePlanningEnd = time.perf_counter_ns()
        self.update_perf(timePlanningStart, timePlanningEnd)
        return path


class RRTStarMulti2D(RRTStarMulti):

    def __init__(self,
                 xStart,
                 xAppList,
                 xGoalList,
                 eta=0.3,
                 subEta=0.05,
                 maxIteration=2000,
                 numDoF=2,
                 envChoice="Planar",
                 nearGoalRadius=0.3,
                 rewireRadius=None,
                 endIterationID=1,
                 print_debug=False):
        super().__init__(xStart=xStart,
                         xAppList=xAppList,
                         xGoalList=xGoalList,
                         eta=eta,
                         subEta=subEta,
                         maxIteration=maxIteration,
                         numDoF=numDoF,
                         envChoice=envChoice,
                         nearGoalRadius=nearGoalRadius,
                         rewireRadius=rewireRadius,
                         endIterationID=endIterationID,
                         print_debug=print_debug)

    def planning(self):
        timePlanningStart = time.perf_counter_ns()
        self.start()
        path = self.get_path()
        timePlanningEnd = time.perf_counter_ns()
        self.update_perf(timePlanningStart, timePlanningEnd)
        return path


class RRTInformedMulti2D(RRTInformedMulti):

    def __init__(self,
                 xStart,
                 xAppList,
                 xGoalList,
                 eta=0.3,
                 subEta=0.05,
                 maxIteration=2000,
                 numDoF=2,
                 envChoice="Planar",
                 nearGoalRadius=0.3,
                 rewireRadius=None,
                 endIterationID=1,
                 print_debug=False):
        super().__init__(xStart=xStart,
                         xAppList=xAppList,
                         xGoalList=xGoalList,
                         eta=eta,
                         subEta=subEta,
                         maxIteration=maxIteration,
                         numDoF=numDoF,
                         envChoice=envChoice,
                         nearGoalRadius=nearGoalRadius,
                         rewireRadius=rewireRadius,
                         endIterationID=endIterationID,
                         print_debug=print_debug)

    def planning(self):
        timePlanningStart = time.perf_counter_ns()
        self.start()
        path = self.get_path()
        timePlanningEnd = time.perf_counter_ns()
        self.update_perf(timePlanningStart, timePlanningEnd)
        return path


class RRTStarConnectMulti2D(RRTStarConnectMulti):

    def __init__(self,
                 xStart,
                 xAppList,
                 xGoalList,
                 eta=0.3,
                 subEta=0.05,
                 maxIteration=2000,
                 numDoF=2,
                 envChoice="Planar",
                 nearGoalRadius=None,
                 rewireRadius=None,
                 endIterationID=1,
                 print_debug=False,
                 localOptEnable=False):
        super().__init__(xStart=xStart,
                         xAppList=xAppList,
                         xGoalList=xGoalList,
                         eta=eta,
                         subEta=subEta,
                         maxIteration=maxIteration,
                         numDoF=numDoF,
                         envChoice=envChoice,
                         nearGoalRadius=nearGoalRadius,
                         rewireRadius=rewireRadius,
                         endIterationID=endIterationID,
                         print_debug=print_debug, 
                         localOptEnable=localOptEnable)

    def planning(self):
        timePlanningStart = time.perf_counter_ns()
        self.start()
        path = self.get_path()
        timePlanningEnd = time.perf_counter_ns()
        self.update_perf(timePlanningStart, timePlanningEnd)
        return path


if __name__ == "__main__":
    np.random.seed(9)  # 922
    import matplotlib.pyplot as plt
    # import seaborn as sns
    # sns.set_theme()
    # sns.set_context("paper")
    from util.general_util import write_dict_to_file

    xStart = np.array([0, 0]).reshape(2, 1)
    xAppList = [np.array([np.pi / 2 - 0.1, 0.2]).reshape(2, 1), np.array([1.45, -0.191]).reshape(2, 1), np.array([1.73, -0.160]).reshape(2, 1)]
    xGoalList = [np.array([np.pi / 2, 0]).reshape(2, 1), np.array([np.pi / 2, 0]).reshape(2, 1), np.array([np.pi / 2, 0]).reshape(2, 1)]

    # planner = RRTBaseMulti2D(xStart, xAppList, xGoalList, eta=0.3, maxIteration=3000)
    # planner = RRTConnectMulti2D(xStart, xAppList, xGoalList, eta=0.3, maxIteration=3000, localOptEnable=True)
    # planner = RRTConnectMulti2D(xStart, xAppList, xGoalList, eta=0.3, maxIteration=3000, localOptEnable=False)
    # planner = RRTStarMulti2D(xStart, xAppList, xGoalList, eta=0.3, maxIteration=3000)
    # planner = RRTInformedMulti2D(xStart, xAppList, xGoalList, eta=0.3, maxIteration=3000)
    planner = RRTStarConnectMulti2D(xStart, xAppList, xGoalList, eta=0.3, maxIteration=3000, localOptEnable=True)
    # planner = RRTStarConnectMulti2D(xStart, xAppList, xGoalList, eta=0.3, maxIteration=3000, localOptEnable=False)

    # planner.robotEnv.robot.plot_arm(xStart, plt_basis=True)
    # planner.robotEnv.robot.plot_arm(xGoalList)
    # planner.robotEnv.robot.plot_arm(xAppList)
    # for obs in planner.robotEnv.taskMapObs:
    #     obs.plot()
    # plt.show()

    path = planner.planning()
    print(f"==>> path: \n{path}")
    print(planner.perfMatrix)
    # write_dict_to_file(planner.perfMatrix, "./planner/result_2d/result_2d_rrtinformed.txt")
    fig, ax = plt.subplots()
    fig.set_size_inches(w=3.40067, h=3.40067)
    fig.tight_layout()
    plt.xlim((-np.pi, np.pi))
    plt.ylim((-np.pi, np.pi))
    planner.plot_tree(path, ax)
    plt.show()

    # plt.axes().set_aspect('equal')
    # plt.axvline(x=0, c="green")
    # plt.axhline(y=0, c="green")
    # obs_list = planner.robotEnv.taskMapObs
    # for obs in obs_list:
    #     obs.plot()
    # for i in range(len(path)):
    #     planner.robotEnv.robot.plot_arm(path[i].config)
    #     plt.pause(0.3)
    # plt.show()

    fig, ax = plt.subplots()
    planner.plot_performance(ax)
    plt.show()