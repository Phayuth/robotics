import os
import sys

wd = os.path.abspath(os.getcwd())
sys.path.append(str(wd))

import time
import matplotlib.pyplot as plt
import numpy as np
from planner_dev.planner2d.planar_rrt_single import RRTConnectDev2D, RRTStarDev2D, RRTInformedDev2D, RRTStarConnectDev2D, RRTInformedConnectDev2D, RRTConnectAstInformedDev2D, RRTConnectLocalOpt2D, RRTStarLocalOpt2D
from planner_dev.copsim6d.copsim_rrt_single import RRTConnectDevCopSim6D, RRTStarDevCopSim6D, RRTInformedDevCopSim6D, RRTStarConnectDevCopSim6D, RRTInformedConnectDevCopSim6D, RRTConnectAstInformedDevCopSim6D, RRTConnectLocalOptCopSim6D, RRTStarLocalOptCopSim6D
from target_localization.pre_record_value import wrap_to_pi, newThetaInit, newThetaApp, newThetaGoal

# # # multi goal
# # xApp = [np.array([np.pi/2-0.1, 0.2]).reshape(2, 1),
# #         np.array([1.45, -0.191]).reshape(2, 1),
# #         np.array([1.73, -0.160]).reshape(2, 1)]

# # xGoal = [np.array([np.pi/2, 0]).reshape(2, 1),
# #          np.array([np.pi/2, 0]).reshape(2, 1),
# #          np.array([np.pi/2, 0]).reshape(2, 1)]


class TrialRun:

    def __init__(self, xStart, xGoal, xApp, eta, maxIteration) -> None:
        self.xStart = xStart
        self.xGoal = xGoal
        self.xApp = xApp
        self.eta = eta
        self.maxIteration = maxIteration

        # self.plannerList = [RRTConnectDev2D, RRTStarDev2D, RRTInformedDev2D, RRTStarConnectDev2D, RRTInformedConnectDev2D, RRTConnectAstInformedDev2D, RRTConnectLocalOpt2D, RRTStarLocalOpt2D]
        self.plannerList = [RRTConnectDevCopSim6D, RRTStarDevCopSim6D, RRTInformedDevCopSim6D, RRTStarConnectDevCopSim6D, RRTInformedConnectDevCopSim6D, RRTConnectAstInformedDevCopSim6D, RRTConnectLocalOptCopSim6D, RRTStarLocalOptCopSim6D]
        self.columnDataName = [plannerClass.__name__ for plannerClass in self.plannerList]
        self.rowDataName = [
            '# Node', 
            'Initial Path Cost', 
            'Init Path Found on Itera', 
            'Final Path Cost', 
            'Full Planning Time (sec)', 
            'Planning Time Only (sec)', 
            'Collision Check Time (sec)', 
            '# Collision Check',
            'Avg Col. Check Time (sec)']

        self.numTrial = 10
        self.plannerData = []
        self.dataMeanVar = None

    def run(self):
        for plannerId in range(len(self.plannerList)):
            numberOfNode = []
            initialCost = []
            initialCostFoundonIteration = []
            finalCost = []
            fullPlanningTime = []
            planningTimeOnly = []
            kCDCheckTime = []
            numberOfCollisionCheck = []
            avgKCDTime = []

            for i in range(self.numTrial):
                np.random.seed(i + 1)
                time.sleep(2)
                planner = self.plannerList[plannerId](self.xStart, self.xApp, self.xGoal, eta=self.eta, maxIteration=self.maxIteration)
                path = planner.planning()
                print(planner.perfMatrix)

                numberOfNode.append(planner.perfMatrix["Number of Node"])
                initialCost.append(planner.perfMatrix["Cost Graph"][0][1])
                initialCostFoundonIteration.append(planner.perfMatrix["Cost Graph"][0][0])
                finalCost.append(planner.perfMatrix["Cost Graph"][-1][1])
                fullPlanningTime.append(planner.perfMatrix["Total Planning Time"])
                planningTimeOnly.append(planner.perfMatrix["Planning Time Only"])
                kCDCheckTime.append(planner.perfMatrix["KCD Time Spend"])
                numberOfCollisionCheck.append(planner.perfMatrix["Number of Collision Check"])
                avgKCDTime.append(planner.perfMatrix["Average KCD Time"])

            self.plannerData.append([numberOfNode, initialCost, initialCostFoundonIteration, finalCost, fullPlanningTime, planningTimeOnly, kCDCheckTime, numberOfCollisionCheck, avgKCDTime])

        data = [[f"{np.mean(trialListData):.4f} $\pm$ {np.std(trialListData):.4f}" for trialListData in allMatricPerPlanner] for allMatricPerPlanner in self.plannerData]
        self.dataMeanVar = np.array(data).T

    def plot(self, ax):
        table = ax.table(rowLabels=self.rowDataName, colLabels=self.columnDataName, cellText=self.dataMeanVar, loc='center')

        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.5)
        table.auto_set_column_width(range(len(self.columnDataName)))
        ax.axis('off')
        ax.set_title(f'Performance of Sampling Based, eta [{self.eta}], maxIteration [{self.maxIteration}], trial run [{self.numTrial}]')


if __name__ == "__main__":
    # 2D
    # xStart = np.array([0, 0]).reshape(2, 1)
    # xGoal = np.array([np.pi / 2, 0]).reshape(2, 1)
    # xApp = np.array([np.pi / 2 - 0.1, 0.2]).reshape(2, 1)

    # 6D
    xStart = newThetaInit
    xGoal = wrap_to_pi(newThetaGoal)
    xApp = wrap_to_pi(newThetaApp)

    eta = 0.1
    maxIteration = 3000

    trun = TrialRun(xStart, xGoal, xApp, eta, maxIteration)
    trun.run()

    fig, ax = plt.subplots()
    trun.plot(ax)
    plt.show()
