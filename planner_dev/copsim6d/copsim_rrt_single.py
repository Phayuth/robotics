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


class RRTBaseCopSim(RRTBase):

    def __init__(self, xStart, xApp, xGoal, eta=0.3, subEta=0.05, maxIteration=2000, numDoF=6, envChoice="CopSim", nearGoalRadius=0.3):
        super().__init__(xStart=xStart, xApp=xApp, xGoal=xGoal, eta=eta, subEta=subEta, maxIteration=maxIteration, numDoF=numDoF, envChoice=envChoice, nearGoalRadius=nearGoalRadius)

    def planning(self):
        self.robotEnv.start_sim()

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

        self.robotEnv.stop_sim()
        return path


class RRTConnectCopSim(RRTConnect):

    def __init__(self, xStart, xApp, xGoal, eta=0.3, subEta=0.05, maxIteration=2000, numDoF=6, envChoice="CopSim"):
        super().__init__(xStart=xStart, xApp=xApp, xGoal=xGoal, eta=eta, subEta=subEta, maxIteration=maxIteration, numDoF=numDoF, envChoice=envChoice)

    def planning(self):
        self.robotEnv.start_sim()

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

        self.robotEnv.stop_sim()
        return path


class RRTStarCopSim(RRTStar):

    def __init__(self, xStart, xApp, xGoal, eta=0.3, subEta=0.05, maxIteration=2000, numDoF=6, envChoice="CopSim", nearGoalRadius=0.3):
        super().__init__(xStart=xStart, xApp=xApp, xGoal=xGoal, eta=eta, subEta=subEta, maxIteration=maxIteration, numDoF=numDoF, envChoice=envChoice, nearGoalRadius=nearGoalRadius)

    def planning(self):
        self.robotEnv.start_sim()

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

        self.robotEnv.stop_sim()
        return path


class RRTInformedCopSim(RRTInformed):

    def __init__(self, xStart, xApp, xGoal, eta=0.3, subEta=0.05, maxIteration=2000, numDoF=6, envChoice="CopSim", nearGoalRadius=0.3):
        super().__init__(xStart=xStart, xApp=xApp, xGoal=xGoal, eta=eta, subEta=subEta, maxIteration=maxIteration, numDoF=numDoF, envChoice=envChoice, nearGoalRadius=nearGoalRadius)

    def planning(self):
        self.robotEnv.start_sim()

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

        self.robotEnv.stop_sim()
        return path


class RRTStarConnectCopSim(RRTStarConnect):

    def __init__(self, xStart, xApp, xGoal, eta=0.3, subEta=0.05, maxIteration=2000, numDoF=6, envChoice="CopSim"):
        super().__init__(xStart=xStart, xApp=xApp, xGoal=xGoal, eta=eta, subEta=subEta, maxIteration=maxIteration, numDoF=numDoF, envChoice=envChoice)

    def planning(self):
        self.robotEnv.start_sim()

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

        self.robotEnv.stop_sim()
        return path


class RRTInformedConnectCopSim(RRTInformedConnect):

    def __init__(self, xStart, xApp, xGoal, eta=0.3, subEta=0.05, maxIteration=2000, numDoF=6, envChoice="CopSim"):
        super().__init__(xStart=xStart, xApp=xApp, xGoal=xGoal, eta=eta, subEta=subEta, maxIteration=maxIteration, numDoF=numDoF, envChoice=envChoice)

    def planning(self):
        self.robotEnv.start_sim()

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

        self.robotEnv.stop_sim()
        return path


class RRTConnectAstInformedCopSim(RRTConnectAstInformed):

    def __init__(self, xStart, xApp, xGoal, eta=0.3, subEta=0.05, maxIteration=2000, numDoF=6, envChoice="CopSim", nearGoalRadius=0.3):
        super().__init__(xStart=xStart, xApp=xApp, xGoal=xGoal, eta=eta, subEta=subEta, maxIteration=maxIteration, numDoF=numDoF, envChoice=envChoice, nearGoalRadius=nearGoalRadius)

    def planning(self):
        self.robotEnv.start_sim()

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

        self.robotEnv.stop_sim()
        return path


class RRTStarLocalOptCopSim(RRTStarLocalOpt):

    def __init__(self, xStart, xApp, xGoal, eta=0.3, subEta=0.05, maxIteration=2000, numDoF=6, envChoice="CopSim", nearGoalRadius=0.3):
        super().__init__(xStart=xStart, xApp=xApp, xGoal=xGoal, eta=eta, subEta=subEta, maxIteration=maxIteration, numDoF=numDoF, envChoice=envChoice, nearGoalRadius=nearGoalRadius)

    def planning(self):
        self.robotEnv.start_sim()

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

        self.robotEnv.stop_sim()
        return path


class RRTConnectLocalOptCopSim(RRTConnectLocalOpt):

    def __init__(self, xStart, xApp, xGoal, eta=0.3, subEta=0.05, maxIteration=2000, numDoF=6, envChoice="CopSim", nearGoalRadius=0.3):
        super().__init__(xStart=xStart, xApp=xApp, xGoal=xGoal, eta=eta, subEta=subEta, maxIteration=maxIteration, numDoF=numDoF, envChoice=envChoice, nearGoalRadius=nearGoalRadius)

    def planning(self):
        self.robotEnv.start_sim()

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

        self.robotEnv.stop_sim()
        return path


if __name__ == "__main__":
    np.random.seed(0)
    import matplotlib.pyplot as plt
    from target_localization.pre_record_value import wrap_to_pi, newThetaInit, newThetaApp, newThetaGoal, new3dofThetaInit, new3dofThetaApp, new3dofThetaGoal
    from util.general_util import write_dict_to_file

    # 6DoF
    # numDoF = 6
    # thetaInit = newThetaInit
    # thetaGoal = wrap_to_pi(newThetaGoal)
    # thetaApp = wrap_to_pi(newThetaApp)

    # 3DoF
    numDoF = 3
    thetaInit = new3dofThetaInit
    thetaGoal = wrap_to_pi(new3dofThetaGoal)
    thetaApp = wrap_to_pi(new3dofThetaApp)

    maxIteration = 3000
    eta = 0.15
    # planner = RRTBaseCopSim(thetaInit, thetaApp, thetaGoal, eta=eta, maxIteration=maxIteration, numDoF=numDoF)
    # planner = RRTConnectCopSim(thetaInit, thetaApp, thetaGoal, eta=eta, maxIteration=maxIteration, numDoF=numDoF)
    # planner = RRTStarCopSim(thetaInit, thetaApp, thetaGoal, eta=eta, maxIteration=maxIteration, numDoF=numDoF)
    # planner = RRTInformedCopSim(thetaInit, thetaApp, thetaGoal, eta=eta, maxIteration=maxIteration, numDoF=numDoF)
    # planner = RRTStarConnectCopSim(thetaInit, thetaApp, thetaGoal, eta=eta, maxIteration=maxIteration, numDoF=numDoF)
    # planner = RRTInformedConnectCopSim(thetaInit, thetaApp, thetaGoal, eta=eta, maxIteration=maxIteration, numDoF=numDoF)
    # planner = RRTConnectAstInformedCopSim(thetaInit, thetaApp, thetaGoal, eta=eta, maxIteration=maxIteration, numDoF=numDoF)
    # planner = RRTStarLocalOptCopSim(thetaInit, thetaApp, thetaGoal, eta=eta, maxIteration=maxIteration, numDoF=numDoF)
    planner = RRTConnectLocalOptCopSim(thetaInit, thetaApp, thetaGoal, eta=eta, maxIteration=maxIteration, numDoF=numDoF)

    path = planner.planning()
    print(planner.perfMatrix)
    # write_dict_to_file(planner.perfMatrix, "./planner_dev/result_6d/result_6d_rrtstar.txt")
    print(f"==>> path: \n{path}")

    time.sleep(3)

    # play back
    planner.robotEnv.start_sim()
    planner.robotEnv.set_goal_joint_value(thetaGoal)
    planner.robotEnv.set_aux_joint_value(thetaApp)
    planner.robotEnv.set_start_joint_value(thetaInit)
    for i in range(len(path)):
        planner.robotEnv.set_joint_value(path[i])
        time.sleep(0.3)
        # triggers next simulation step
        # client.step()

    # stop simulation
    planner.robotEnv.stop_sim()

    fig, ax = plt.subplots()
    planner.plot_performance(ax)
    plt.show()