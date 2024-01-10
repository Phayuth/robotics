import os
import sys

wd = os.path.abspath(os.getcwd())
sys.path.append(str(wd))

import time
import matplotlib.pyplot as plt
import numpy as np
import pickle
from planner.rrt_connect import RRTConnect
from planner.rrt_informed import RRTInformed
from planner.rrt_star_connect import RRTStarConnect

from planner.rrt_connect import RRTConnectMulti
from planner.rrt_star_connect import RRTStarConnectMulti

from datasave.joint_value.experiment_paper import ICRABarnMap


class TrialRun:

    def __init__(self, numTrial, plannerName) -> None:
        # q = ICRABarnMap.PoseSingle()
        q = ICRABarnMap.PoseMulti3()
        self.xStart = q.xStart
        self.xApp = q.xApp
        self.xGoal = q.xGoal

        self.eta = 0.3
        self.subEta = 0.05
        self.maxIteration = 2000
        self.numDoF = 2
        self.envChoice = "Planar"
        self.nearGoalRadiusSingleTree = 0.3
        self.nearGoalRadiusDualTree = None
        self.rewireRadius = None
        self.endIterationID = 1
        self.printDebug = False
        self.localOptEnable = True

        self.plannerName = plannerName
        self.perf = ['# Node',
                     'First Path Cost',
                     'First Path Iteration',
                     'Final Path Cost',
                     'Final Path Iteration'
                     'Running Time (sec)',
                     'Planner Time (sec)',
                     'Col. Check Time (sec)',
                     '# Col. Check',
                     'Avg Col. Check Time (sec)']
        self.plannerData = []
        self.costGraph = []
        self.numTrial = numTrial
        self.recordToPaper = []

    def run(self):
        for i in range(self.numTrial):
            np.random.seed(i + 1)
            print(f"Trail {i}-th")
            # time.sleep(2)
            # single
            # planner = RRTInformed(self.xStart, self.xApp, self.xGoal, self.eta, self.subEta, self.maxIteration, self.numDoF, self.envChoice, self.nearGoalRadiusSingleTree, self.rewireRadius, self.endIterationID, self.printDebug)
            # planner = RRTConnect(self.xStart, self.xApp, self.xGoal, self.eta, self.subEta, self.maxIteration, self.numDoF, self.envChoice, self.nearGoalRadiusDualTree, self.rewireRadius, self.endIterationID, self.printDebug, self.localOptEnable)
            # planner = RRTStarConnect(self.xStart, self.xApp, self.xGoal, self.eta, self.subEta, self.maxIteration, self.numDoF, self.envChoice, self.nearGoalRadiusSingleTree, self.rewireRadius, self.endIterationID, self.printDebug, self.localOptEnable)

            # multi
            # planner = RRTConnectMulti(self.xStart, self.xApp, self.xGoal, self.eta, self.subEta, self.maxIteration, self.numDoF, self.envChoice, self.nearGoalRadiusDualTree, self.rewireRadius, self.endIterationID, self.printDebug, self.localOptEnable)
            planner = RRTStarConnectMulti(self.xStart, self.xApp, self.xGoal, self.eta, self.subEta, self.maxIteration, self.numDoF, self.envChoice, self.nearGoalRadiusDualTree, self.rewireRadius, self.endIterationID, self.printDebug, self.localOptEnable)

            # start plan
            timePlanningStart = time.perf_counter_ns()
            planner.start()
            path = planner.get_path()
            timePlanningEnd = time.perf_counter_ns()
            planner.update_perf(timePlanningStart, timePlanningEnd)

            data = [planner.perfMatrix["Number of Node"],
                    planner.perfMatrix["Cost Graph"][0][1],
                    planner.perfMatrix["Cost Graph"][0][0],
                    planner.perfMatrix["Cost Graph"][-1][1],
                    planner.perfMatrix["Cost Graph"][-1][0],
                    planner.perfMatrix["Total Planning Time"],
                    planner.perfMatrix["Planning Time Only"],
                    planner.perfMatrix["KCD Time Spend"],
                    planner.perfMatrix["Number of Collision Check"],
                    planner.perfMatrix["Average KCD Time"]]

            self.plannerData.append(data)
            self.costGraph.append(planner.perfMatrix["Cost Graph"])

        print("Finished Trail Loop")

        plannerData = np.array(self.plannerData)

        # save data
        np.save(f'./datasave/planner_performance/{self.plannerName}.npy', plannerData)
        fileName = f"./datasave/planner_performance/{self.plannerName}_costgraph.pkl"
        with open(fileName, "wb") as file:
            pickle.dump(self.costGraph, file)

    # def plot(self, ax):
    #     table = ax.table(rowLabels=self.rowDataName, colLabels=self.columnDataName, cellText=self.dataMeanVar, loc='center')

    #     table.auto_set_font_size(False)
    #     table.set_fontsize(10)
    #     table.scale(1, 1.5)
    #     table.auto_set_column_width(range(len(self.perf)))
    #     ax.axis('off')
    #     ax.set_title(f'Performance of Sampling Based, eta [{self.eta}], maxIteration [{self.maxIteration}], trial run [{self.numTrial}]')

    def calculate(self):
        # load data
        plannerData = np.load(f'./datasave/planner_performance/{self.plannerName}.npy')
        with open(f"./datasave/planner_performance/{self.plannerName}_costgraph.pkl", "rb") as file:
            loadedList = pickle.load(file)

        print("Calculating Data ...")
        for i in range(plannerData.shape[1]):
            mv = plannerData[:,i]
            self.recordToPaper.append(f"{np.mean(mv)} +- {np.std(mv)}")

        print("Record this to paper")
        print(self.recordToPaper)

if __name__ == "__main__":
    plannerName = "multi_connectstarlocgap2d"
    trun = TrialRun(100, plannerName)
    trun.run()
    trun.calculate()
    # fig, ax = plt.subplots()
    # trun.plot(ax)
    # plt.show()
