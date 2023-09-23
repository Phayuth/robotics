import os
import sys

wd = os.path.abspath(os.getcwd())
sys.path.append(str(wd))

import time
import numpy as np
from planner_dev.rrt_base import RRTBase
from planner_dev.rrt_connect import RRTConnect
from planner_dev.rrt_star import RRTStar
from planner_dev.rrt_informed import RRTInformed
from planner_dev.rrt_star_connect import RRTStarConnect
from planner_dev.rrt_informed_connect import RRTInformedConnect

from planner_dev.rrt_connect_ast_informed import RRTConnectAstInformed
from planner_dev.rrt_star_localopt import RRTStarLocalOpt
from planner_dev.rrt_connect_localopt import RRTConnectLocalOpt


class RRTBaseDev2D(RRTBase):

    def __init__(self, xStart, xApp, xGoal, eta=0.3, subEta=0.05, maxIteration=2000, numDoF=2, envChoice="Planar", nearGoalRadius=0.3):
        super().__init__(xStart=xStart, xApp=xApp, xGoal=xGoal, eta=eta, subEta=subEta, maxIteration=maxIteration, numDoF=numDoF, envChoice=envChoice, nearGoalRadius=nearGoalRadius)

    def planning(self):
        timePlanningStart = time.perf_counter_ns()
        itera = self.start()
        path = self.search_best_cost_singledirection_path(backFromNode=self.xApp, treeVertexList=self.XInGoalRegion, attachNode=self.xGoal)
        timePlanningEnd = time.perf_counter_ns()

        # record performance
        self.perfMatrix["Planner Name"] = f"{self.__class__.__name__}"
        self.perfMatrix["Parameters"]["eta"] = self.eta
        self.perfMatrix["Parameters"]["subEta"] = self.subEta
        self.perfMatrix["Parameters"]["Max Iteration"] = self.maxIteration
        self.perfMatrix["Parameters"]["Rewire Radius"] = 0.0
        self.perfMatrix["Number of Node"] = len(self.treeVertex)
        self.perfMatrix["Total Planning Time"] = (timePlanningEnd-timePlanningStart) * 1e-9
        self.perfMatrix["KCD Time Spend"] = self.perfMatrix["KCD Time Spend"] * 1e-9
        self.perfMatrix["Planning Time Only"] = self.perfMatrix["Total Planning Time"] - self.perfMatrix["KCD Time Spend"]
        self.perfMatrix["Average KCD Time"] = self.perfMatrix["KCD Time Spend"] / self.perfMatrix["Number of Collision Check"]

        return path


class RRTConnect2D(RRTConnect):

    def __init__(self, xStart, xApp, xGoal, eta=0.3, subEta=0.05, maxIteration=2000, numDoF=2, envChoice="Planar"):
        super().__init__(xStart=xStart, xApp=xApp, xGoal=xGoal, eta=eta, subEta=subEta, maxIteration=maxIteration, numDoF=numDoF, envChoice=envChoice)

    def planning(self):
        timePlanningStart = time.perf_counter_ns()
        itera = self.start()
        path = self.search_backtrack_single_directional_path(backFromNode=self.xApp, attachNode=self.xGoal)
        timePlanningEnd = time.perf_counter_ns()

        # record performance
        self.perfMatrix["Planner Name"] = f"{self.__class__.__name__}"
        self.perfMatrix["Parameters"]["eta"] = self.eta
        self.perfMatrix["Parameters"]["subEta"] = self.subEta
        self.perfMatrix["Parameters"]["Max Iteration"] = self.maxIteration
        self.perfMatrix["Parameters"]["Rewire Radius"] = 0.0
        self.perfMatrix["Number of Node"] = len(self.treeVertexStart) + len(self.treeVertexGoal)
        self.perfMatrix["Total Planning Time"] = (timePlanningEnd-timePlanningStart) * 1e-9
        self.perfMatrix["KCD Time Spend"] = self.perfMatrix["KCD Time Spend"] * 1e-9
        self.perfMatrix["Planning Time Only"] = self.perfMatrix["Total Planning Time"] - self.perfMatrix["KCD Time Spend"]
        self.perfMatrix["Average KCD Time"] = self.perfMatrix["KCD Time Spend"] / self.perfMatrix["Number of Collision Check"]

        return path


class RRTStar2D(RRTStar):

    def __init__(self, xStart, xApp, xGoal, eta=0.3, subEta=0.05, maxIteration=2000, numDoF=2, envChoice="Planar", nearGoalRadius=0.3):
        super().__init__(xStart=xStart, xApp=xApp, xGoal=xGoal, eta=eta, subEta=subEta, maxIteration=maxIteration, numDoF=numDoF, envChoice=envChoice, nearGoalRadius=nearGoalRadius)

    def planning(self):
        timePlanningStart = time.perf_counter_ns()
        itera = self.start()
        path = self.search_best_cost_singledirection_path(backFromNode=self.xApp, treeVertexList=self.XInGoalRegion, attachNode=self.xGoal)
        timePlanningEnd = time.perf_counter_ns()

        # record performance
        self.perfMatrix["Planner Name"] = f"{self.__class__.__name__}"
        self.perfMatrix["Parameters"]["eta"] = self.eta
        self.perfMatrix["Parameters"]["subEta"] = self.subEta
        self.perfMatrix["Parameters"]["Max Iteration"] = self.maxIteration
        self.perfMatrix["Parameters"]["Rewire Radius"] = self.rewireRadius
        self.perfMatrix["Number of Node"] = len(self.treeVertex)
        self.perfMatrix["Total Planning Time"] = (timePlanningEnd-timePlanningStart) * 1e-9
        self.perfMatrix["KCD Time Spend"] = self.perfMatrix["KCD Time Spend"] * 1e-9
        self.perfMatrix["Planning Time Only"] = self.perfMatrix["Total Planning Time"] - self.perfMatrix["KCD Time Spend"]
        self.perfMatrix["Average KCD Time"] = self.perfMatrix["KCD Time Spend"] / self.perfMatrix["Number of Collision Check"]

        return path


class RRTInformed2D(RRTInformed):

    def __init__(self, xStart, xApp, xGoal, eta=0.3, subEta=0.05, maxIteration=2000, numDoF=2, envChoice="Planar", nearGoalRadius=0.3):
        super().__init__(xStart=xStart, xApp=xApp, xGoal=xGoal, eta=eta, subEta=subEta, maxIteration=maxIteration, numDoF=numDoF, envChoice=envChoice, nearGoalRadius=nearGoalRadius)

    def planning(self):
        timePlanningStart = time.perf_counter_ns()
        itera = self.start()
        path = self.search_best_cost_singledirection_path(backFromNode=self.xApp, treeVertexList=self.XSoln, attachNode=self.xGoal)
        timePlanningEnd = time.perf_counter_ns()

        # record performance
        self.perfMatrix["Planner Name"] = f"{self.__class__.__name__}"
        self.perfMatrix["Parameters"]["eta"] = self.eta
        self.perfMatrix["Parameters"]["subEta"] = self.subEta
        self.perfMatrix["Parameters"]["Max Iteration"] = self.maxIteration
        self.perfMatrix["Parameters"]["Rewire Radius"] = self.rewireRadius
        self.perfMatrix["Number of Node"] = len(self.treeVertex)
        self.perfMatrix["Total Planning Time"] = (timePlanningEnd-timePlanningStart) * 1e-9
        self.perfMatrix["KCD Time Spend"] = self.perfMatrix["KCD Time Spend"] * 1e-9
        self.perfMatrix["Planning Time Only"] = self.perfMatrix["Total Planning Time"] - self.perfMatrix["KCD Time Spend"]
        self.perfMatrix["Average KCD Time"] = self.perfMatrix["KCD Time Spend"] / self.perfMatrix["Number of Collision Check"]

        return path


class RRTStarConnect2D(RRTStarConnect):

    def __init__(self, xStart, xApp, xGoal, eta=0.3, subEta=0.05, maxIteration=2000, numDoF=2, envChoice="Planar"):
        super().__init__(xStart=xStart, xApp=xApp, xGoal=xGoal, eta=eta, subEta=subEta, maxIteration=maxIteration, numDoF=numDoF, envChoice=envChoice)

    def planning(self):
        timePlanningStart = time.perf_counter_ns()
        itera = self.start()
        path = self.search_best_cost_bidirection_path(connectNodePairList=self.connectNodePair, attachNode=self.xGoal)
        timePlanningEnd = time.perf_counter_ns()

        # record performance
        self.perfMatrix["Planner Name"] = f"{self.__class__.__name__}"
        self.perfMatrix["Parameters"]["eta"] = self.eta
        self.perfMatrix["Parameters"]["subEta"] = self.subEta
        self.perfMatrix["Parameters"]["Max Iteration"] = self.maxIteration
        self.perfMatrix["Parameters"]["Rewire Radius"] = self.rewireRadius
        self.perfMatrix["Number of Node"] = len(self.treeVertexStart) + len(self.treeVertexGoal)
        self.perfMatrix["Total Planning Time"] = (timePlanningEnd-timePlanningStart) * 1e-9
        self.perfMatrix["KCD Time Spend"] = self.perfMatrix["KCD Time Spend"] * 1e-9
        self.perfMatrix["Planning Time Only"] = self.perfMatrix["Total Planning Time"] - self.perfMatrix["KCD Time Spend"]
        self.perfMatrix["Average KCD Time"] = self.perfMatrix["KCD Time Spend"] / self.perfMatrix["Number of Collision Check"]

        return path


class RRTInformedConnect2D(RRTInformedConnect):

    def __init__(self, xStart, xApp, xGoal, eta=0.3, subEta=0.05, maxIteration=2000, numDoF=2, envChoice="Planar"):
        super().__init__(xStart=xStart, xApp=xApp, xGoal=xGoal, eta=eta, subEta=subEta, maxIteration=maxIteration, numDoF=numDoF, envChoice=envChoice)

    def planning(self):
        timePlanningStart = time.perf_counter_ns()
        itera = self.start()
        path = self.search_best_cost_bidirection_path(connectNodePairList=self.connectNodePair, attachNode=self.xGoal)
        timePlanningEnd = time.perf_counter_ns()

        # record performance
        self.perfMatrix["Planner Name"] = f"{self.__class__.__name__}"
        self.perfMatrix["Parameters"]["eta"] = self.eta
        self.perfMatrix["Parameters"]["subEta"] = self.subEta
        self.perfMatrix["Parameters"]["Max Iteration"] = self.maxIteration
        self.perfMatrix["Parameters"]["Rewire Radius"] = self.rewireRadius
        self.perfMatrix["Number of Node"] = len(self.treeVertexStart) + len(self.treeVertexGoal)
        self.perfMatrix["Total Planning Time"] = (timePlanningEnd-timePlanningStart) * 1e-9
        self.perfMatrix["KCD Time Spend"] = self.perfMatrix["KCD Time Spend"] * 1e-9
        self.perfMatrix["Planning Time Only"] = self.perfMatrix["Total Planning Time"] - self.perfMatrix["KCD Time Spend"]
        self.perfMatrix["Average KCD Time"] = self.perfMatrix["KCD Time Spend"] / self.perfMatrix["Number of Collision Check"]

        return path


class RRTConnectAstInformed2D(RRTConnectAstInformed):

    def __init__(self, xStart, xApp, xGoal, eta=0.3, subEta=0.05, maxIteration=2000, numDoF=2, envChoice="Planar", nearGoalRadius=0.3):
        super().__init__(xStart=xStart, xApp=xApp, xGoal=xGoal, eta=eta, subEta=subEta, maxIteration=maxIteration, numDoF=numDoF, envChoice=envChoice, nearGoalRadius=nearGoalRadius)

    def planning(self):
        timePlanningStart = time.perf_counter_ns()
        itera = self.start()
        path = self.search_best_cost_singledirection_path(backFromNode=self.xApp, treeVertexList=self.XSoln, attachNode=self.xGoal)
        timePlanningEnd = time.perf_counter_ns()

        # record performance
        self.perfMatrix["Planner Name"] = f"{self.__class__.__name__}"
        self.perfMatrix["Parameters"]["eta"] = self.eta
        self.perfMatrix["Parameters"]["subEta"] = self.subEta
        self.perfMatrix["Parameters"]["Max Iteration"] = self.maxIteration
        self.perfMatrix["Parameters"]["Rewire Radius"] = self.rewireRadius
        self.perfMatrix["Number of Node"] = len(self.treeVertexStart)
        self.perfMatrix["Total Planning Time"] = (timePlanningEnd-timePlanningStart) * 1e-9
        self.perfMatrix["KCD Time Spend"] = self.perfMatrix["KCD Time Spend"] * 1e-9
        self.perfMatrix["Planning Time Only"] = self.perfMatrix["Total Planning Time"] - self.perfMatrix["KCD Time Spend"]
        self.perfMatrix["Average KCD Time"] = self.perfMatrix["KCD Time Spend"] / self.perfMatrix["Number of Collision Check"]

        return path


class RRTStarLocalOpt2D(RRTStarLocalOpt):

    def __init__(self, xStart, xApp, xGoal, eta=0.3, subEta=0.05, maxIteration=2000, numDoF=2, envChoice="Planar", nearGoalRadius=0.3):
        super().__init__(xStart=xStart, xApp=xApp, xGoal=xGoal, eta=eta, subEta=subEta, maxIteration=maxIteration, numDoF=numDoF, envChoice=envChoice, nearGoalRadius=nearGoalRadius)

    def planning(self):
        timePlanningStart = time.perf_counter_ns()
        itera = self.start()
        path = self.search_best_cost_singledirection_path(backFromNode=self.xApp, treeVertexList=self.XInGoalRegion, attachNode=self.xGoal)
        timePlanningEnd = time.perf_counter_ns()

        # record performance
        self.perfMatrix["Planner Name"] = f"{self.__class__.__name__}"
        self.perfMatrix["Parameters"]["eta"] = self.eta
        self.perfMatrix["Parameters"]["subEta"] = self.subEta
        self.perfMatrix["Parameters"]["Max Iteration"] = self.maxIteration
        self.perfMatrix["Parameters"]["Rewire Radius"] = self.rewireRadius
        self.perfMatrix["Number of Node"] = len(self.treeVertex)
        self.perfMatrix["Total Planning Time"] = (timePlanningEnd-timePlanningStart) * 1e-9
        self.perfMatrix["KCD Time Spend"] = self.perfMatrix["KCD Time Spend"] * 1e-9
        self.perfMatrix["Planning Time Only"] = self.perfMatrix["Total Planning Time"] - self.perfMatrix["KCD Time Spend"]
        self.perfMatrix["Average KCD Time"] = self.perfMatrix["KCD Time Spend"] / self.perfMatrix["Number of Collision Check"]

        return path


class RRTConnectLocalOpt2D(RRTConnectLocalOpt):

    def __init__(self, xStart, xApp, xGoal, eta=0.3, subEta=0.05, maxIteration=2000, numDoF=2, envChoice="Planar", nearGoalRadius=0.3):
        super().__init__(xStart=xStart, xApp=xApp, xGoal=xGoal, eta=eta, subEta=subEta, maxIteration=maxIteration, numDoF=numDoF, envChoice=envChoice, nearGoalRadius=nearGoalRadius)

    def planning(self):
        timePlanningStart = time.perf_counter_ns()
        itera = self.start()
        path = self.search_best_cost_singledirection_path(backFromNode=self.xApp, treeVertexList=self.XInGoalRegion, attachNode=self.xGoal)
        timePlanningEnd = time.perf_counter_ns()

        # record performance
        self.perfMatrix["Planner Name"] = f"{self.__class__.__name__}"
        self.perfMatrix["Parameters"]["eta"] = self.eta
        self.perfMatrix["Parameters"]["subEta"] = self.subEta
        self.perfMatrix["Parameters"]["Max Iteration"] = self.maxIteration
        self.perfMatrix["Parameters"]["Rewire Radius"] = self.rewireRadius
        self.perfMatrix["Number of Node"] = len(self.treeVertexStart)
        self.perfMatrix["Total Planning Time"] = (timePlanningEnd-timePlanningStart) * 1e-9
        self.perfMatrix["KCD Time Spend"] = self.perfMatrix["KCD Time Spend"] * 1e-9
        self.perfMatrix["Planning Time Only"] = self.perfMatrix["Total Planning Time"] - self.perfMatrix["KCD Time Spend"]
        self.perfMatrix["Average KCD Time"] = self.perfMatrix["KCD Time Spend"] / self.perfMatrix["Number of Collision Check"]

        return path


if __name__ == "__main__":
    np.random.seed(9)
    import matplotlib.pyplot as plt
    # import seaborn as sns
    # sns.set_theme()
    # sns.set_context("paper")
    from planner_util.coord_transform import circle_plt
    from util.general_util import write_dict_to_file

    xStart = np.array([0, 0]).reshape(2, 1)
    xGoal = np.array([np.pi / 2, 0]).reshape(2, 1)
    xApp = np.array([np.pi / 2 - 0.1, 0.2]).reshape(2, 1)

    # planner = RRTBase2D(xStart, xApp, xGoal, eta=0.3, maxIteration=3000)
    # planner = RRTConnect2D(xStart, xApp, xGoal, eta=0.3, maxIteration=3000)
    # planner = RRTStar2D(xStart, xApp, xGoal, eta=0.3, maxIteration=3000)
    # planner = RRTInformed2D(xStart, xApp, xGoal, eta=0.3, maxIteration=3000)
    planner = RRTStarConnect2D(xStart, xApp, xGoal, eta=0.3, maxIteration=3000)
    # planner = RRTInformedConnect2D(xStart, xApp, xGoal, eta=0.3, maxIteration=3000)
    # planner = RRTConnectAstInformed2D(xStart, xApp, xGoal, eta=0.3, maxIteration=3000)
    # planner = RRTStarLocalOpt2D(xStart, xApp, xGoal, eta=0.3, maxIteration=3000)
    # planner = RRTConnectLocalOpt2D(xStart, xApp, xGoal, eta=0.3, maxIteration=3000)

    planner.robotEnv.robot.plot_arm(xStart, plt_basis=True)
    planner.robotEnv.robot.plot_arm(xGoal)
    planner.robotEnv.robot.plot_arm(xApp)
    for obs in planner.robotEnv.taskMapObs:
        obs.plot()
    plt.show()

    path = planner.planning()
    print(f"==>> path: \n{path}")
    print(planner.perfMatrix)
    # write_dict_to_file(planner.perfMatrix, "./planner_dev/result_2d/result_2d_rrtinformed.txt")
    fig, ax = plt.subplots()
    fig.set_size_inches(w=3.40067, h=3.40067)
    fig.tight_layout()
    circle_plt(planner.xGoal.config[0, 0], planner.xGoal.config[1, 0], planner.distGoalToApp)
    # circle_plt(planner.xApp.config[0, 0], planner.xApp.config[1, 0], planner.nearGoalRadius)
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