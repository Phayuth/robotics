import os
import sys

wd = os.path.abspath(os.getcwd())
sys.path.append(str(wd))

import time
import numpy as np
from planner_dev.rrt_base import RRTBaseMulti
from planner_dev.rrt_connect import RRTConnectMulti
from planner_dev.rrt_star import RRTStarMulti
from planner_dev.rrt_informed import RRTInformedMulti
from planner_dev.rrt_star_connect import RRTStarConnectMulti


class RRTBaseMultiCopSim(RRTBaseMulti):

    def __init__(self, xStart, xAppList, xGoalList, eta=0.3, subEta=0.05, maxIteration=2000, numDoF=6, envChoice="CopSim", nearGoalRadius=0.3):
        super().__init__(xStart=xStart, xAppList=xAppList, xGoalList=xGoalList, eta=eta, subEta=subEta, maxIteration=maxIteration, numDoF=numDoF, envChoice=envChoice, nearGoalRadius=nearGoalRadius)

    def planning(self):
        self.robotEnv.start_sim()

        timePlanningStart = time.perf_counter_ns()
        itera = self.start()
        path = self.search_best_cost_singledirection_path(backFromNode=self.xAppList[self.xGoalBestIndex], treeVertexList=self.XInGoalRegion[self.xGoalBestIndex], attachNode=self.xGoalList[self.xGoalBestIndex])
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


class RRTConnectMultiCopSim(RRTConnectMulti):

    def __init__(self, xStart, xAppList, xGoalList, eta=0.3, subEta=0.05, maxIteration=2000, numDoF=6, envChoice="CopSim"):
        super().__init__(xStart=xStart, xAppList=xAppList, xGoalList=xGoalList, eta=eta, subEta=subEta, maxIteration=maxIteration, numDoF=numDoF, envChoice=envChoice)

    def planning(self):
        self.robotEnv.start_sim()

        timePlanningStart = time.perf_counter_ns()
        itera = self.start()
        path = self.search_best_cost_bidirection_path(connectNodePairList=self.connectNodePair)
        xGoalIndex = self.xAppList.index(path[-1])
        path = path + [self.xGoalList[xGoalIndex]]
        timePlanningEnd = time.perf_counter_ns()

        # record performance
        self.perfMatrix["Planner Name"] = f"{self.__class__.__name__}"
        self.perfMatrix["Parameters"]["eta"] = self.eta
        self.perfMatrix["Parameters"]["subEta"] = self.subEta
        self.perfMatrix["Parameters"]["Max Iteration"] = self.maxIteration
        self.perfMatrix["Parameters"]["Rewire Radius"] = None
        self.perfMatrix["Number of Node"] = len(self.treeVertexStart) + len(self.treeVertexGoal)
        self.perfMatrix["Total Planning Time"] = (timePlanningEnd-timePlanningStart) * 1e-9
        self.perfMatrix["KCD Time Spend"] = self.perfMatrix["KCD Time Spend"] * 1e-9
        self.perfMatrix["Planning Time Only"] = self.perfMatrix["Total Planning Time"] - self.perfMatrix["KCD Time Spend"]
        self.perfMatrix["Average KCD Time"] = self.perfMatrix["KCD Time Spend"] / self.perfMatrix["Number of Collision Check"]

        self.robotEnv.stop_sim()
        return path
    

class RRTStarMultiCopSim(RRTStarMulti):

    def __init__(self, xStart, xAppList, xGoalList, eta=0.3, subEta=0.05, maxIteration=2000, numDoF=6, envChoice="CopSim", nearGoalRadius=0.3):
        super().__init__(xStart=xStart, xAppList=xAppList, xGoalList=xGoalList, eta=eta, subEta=subEta, maxIteration=maxIteration, numDoF=numDoF, envChoice=envChoice, nearGoalRadius=nearGoalRadius)

    def planning(self):
        self.robotEnv.start_sim()

        timePlanningStart = time.perf_counter_ns()
        itera = self.start()
        path = self.search_best_cost_singledirection_path(backFromNode=self.xAppList[self.xGoalBestIndex], treeVertexList=self.XInGoalRegion[self.xGoalBestIndex], attachNode=self.xGoalList[self.xGoalBestIndex])
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


class RRTInformedMultiCopSim(RRTInformedMulti):

    def __init__(self, xStart, xAppList, xGoalList, eta=0.3, subEta=0.05, maxIteration=2000, numDoF=6, envChoice="CopSim", nearGoalRadius=0.3):
        super().__init__(xStart=xStart, xAppList=xAppList, xGoalList=xGoalList, eta=eta, subEta=subEta, maxIteration=maxIteration, numDoF=numDoF, envChoice=envChoice, nearGoalRadius=nearGoalRadius)

    def planning(self):
        self.robotEnv.start_sim()

        timePlanningStart = time.perf_counter_ns()
        itera = self.start()
        path = self.search_best_cost_singledirection_path(backFromNode=self.xAppList[self.xGoalBestIndex], treeVertexList=self.XSoln[self.xGoalBestIndex], attachNode=self.xGoalList[self.xGoalBestIndex])
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


class RRTStarConnectMultiCopSim(RRTStarConnectMulti):

    def __init__(self, xStart, xAppList, xGoalList, eta=0.3, subEta=0.05, maxIteration=2000, numDoF=6, envChoice="CopSim"):
        super().__init__(xStart=xStart, xAppList=xAppList, xGoalList=xGoalList, eta=eta, subEta=subEta, maxIteration=maxIteration, numDoF=numDoF, envChoice=envChoice)

    def planning(self):
        self.robotEnv.start_sim()

        timePlanningStart = time.perf_counter_ns()
        itera = self.start()
        path = self.search_best_cost_bidirection_path(connectNodePairList=self.connectNodePair)
        xGoalIndex = self.xAppList.index(path[-1])
        path = path + [self.xGoalList[xGoalIndex]]
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


if __name__ == "__main__":
    np.random.seed(0)
    import matplotlib.pyplot as plt
    from target_localization.pre_record_value import wrap_to_pi, newThetaInit, newThetaApp, newThetaGoal
    from util.general_util import write_dict_to_file

    # 6DoF
    numDoF = 6
    thetaInit = newThetaInit
    thetaGoalList = [wrap_to_pi(newThetaGoal), wrap_to_pi(newThetaGoal), wrap_to_pi(newThetaGoal)]
    thetaAppList = [wrap_to_pi(newThetaApp), wrap_to_pi(newThetaApp), wrap_to_pi(newThetaApp)]

    # 3DoF
    # numDoF = 3
    # thetaInit = new3dofThetaInit
    # thetaGoalList = wrap_to_pi(new3dofThetaGoal)
    # thetaAppList = wrap_to_pi(new3dofThetaApp)

    maxIteration = 3000
    eta = 0.15
    # planner = RRTBaseMultiCopSim(thetaInit, thetaAppList, thetaGoalList, eta=eta, maxIteration=maxIteration)
    # planner = RRTConnectMultiCopSim(thetaInit, thetaAppList, thetaGoalList, eta=eta, maxIteration=maxIteration)
    # planner = RRTStarMultiCopSim(thetaInit, thetaAppList, thetaGoalList, eta=eta, maxIteration=maxIteration)
    # planner = RRTInformedMultiCopSim(thetaInit, thetaAppList, thetaGoalList, eta=eta, maxIteration=maxIteration)
    planner = RRTStarConnectMultiCopSim(thetaInit, thetaAppList, thetaGoalList, eta=eta, maxIteration=maxIteration)

    path = planner.planning()
    print(planner.perfMatrix)
    # write_dict_to_file(planner.perfMatrix, "./planner_dev/result_6d/result_6d_rrtstar.txt")
    print(f"==>> path: \n{path}")

    time.sleep(3)

    # play back
    planner.robotEnv.start_sim()
    planner.robotEnv.set_start_joint_value(path[0])
    planner.robotEnv.set_goal_joint_value(path[-1])
    planner.robotEnv.set_aux_joint_value(path[-2])
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