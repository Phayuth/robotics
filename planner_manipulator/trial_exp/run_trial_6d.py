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

from planner_manipulator.joint_process import JointProcess

from datasave.joint_value.experiment_paper import URHarvesting


class TrialRun:

    def __init__(self, numTrial, plannerName) -> None:
        # q = ICRABarnMap.PoseSingle()
        q = URHarvesting.PoseMulti
        self.xStart = q.xStart
        self.xApp = q.xApp
        self.xGoal = q.xGoal

        # process joint
        self.xStart = JointProcess.wrap_to_pi(self.xStart)
        if isinstance(self.xApp, list):
            self.xApp = [JointProcess.wrap_to_pi(x) for x in self.xApp]
            self.xGoal = [JointProcess.wrap_to_pi(x) for x in self.xGoal]
        else:
            self.xApp = JointProcess.wrap_to_pi(self.xApp)
            self.xGoal = JointProcess.wrap_to_pi(self.xGoal)

        self.eta = 0.15
        self.subEta = 0.05
        self.maxIteration = 3000
        self.numDoF = 6
        self.envChoice = "CopSim"
        self.nearGoalRadiusSingleTree = 0.3
        self.nearGoalRadiusDualTree = None
        self.rewireRadius = None
        self.endIterationID = 1
        self.print_debug = True
        self.localOptEnable = False

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
            time.sleep(2)
            # single
            # planner = RRTInformed(self.xStart, self.xApp, self.xGoal, self.eta, self.subEta, self.maxIteration, self.numDoF, self.envChoice, self.nearGoalRadiusSingleTree, self.rewireRadius, self.endIterationID, self.print_debug)
            # planner = RRTConnect(self.xStart, self.xApp, self.xGoal, self.eta, self.subEta, self.maxIteration, self.numDoF, self.envChoice, self.nearGoalRadiusDualTree, self.rewireRadius, self.endIterationID, self.print_debug, self.localOptEnable)
            # planner = RRTStarConnect(self.xStart, self.xApp, self.xGoal, self.eta, self.subEta, self.maxIteration, self.numDoF, self.envChoice, self.nearGoalRadiusSingleTree, self.rewireRadius, self.endIterationID, self.print_debug, self.localOptEnable)

            # multi
            # planner = RRTConnectMulti(self.xStart, self.xApp, self.xGoal, self.eta, self.subEta, self.maxIteration, self.numDoF, self.envChoice, self.nearGoalRadiusDualTree, self.rewireRadius, self.endIterationID, self.print_debug, self.localOptEnable)
            planner = RRTStarConnectMulti(self.xStart, self.xApp, self.xGoal, self.eta, self.subEta, self.maxIteration, self.numDoF, self.envChoice, self.nearGoalRadiusDualTree, self.rewireRadius, self.endIterationID, self.print_debug, self.localOptEnable)


            planner.robotEnvClass.start_sim()

            # start plan
            timePlanningStart = time.perf_counter_ns()
            planner.start()
            path = planner.get_path()
            timePlanningEnd = time.perf_counter_ns()
            planner.update_perf(timePlanningStart, timePlanningEnd)

            planner.robotEnvClass.stop_sim()

            if len(planner.perfMatrix["Cost Graph"])==0:
                data = [planner.perfMatrix["Number of Node"],
                        None,
                        None,
                        None,
                        None,
                        planner.perfMatrix["Total Planning Time"],
                        planner.perfMatrix["Planning Time Only"],
                        planner.perfMatrix["KCD Time Spend"],
                        planner.perfMatrix["Number of Collision Check"],
                        planner.perfMatrix["Average KCD Time"]]
            else:
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

        # plannerData = np.array(self.plannerData)

        # save data
        table = f"./datasave/planner_performance/{self.plannerName}_table.pkl"
        with open(table, "wb") as file:
            pickle.dump(self.plannerData, file)

        fileName = f"./datasave/planner_performance/{self.plannerName}_costgraph.pkl"
        with open(fileName, "wb") as file:
            pickle.dump(self.costGraph, file)

    # def calculate(self):
    #     # load data
    #     plannerData = np.load(f'./datasave/planner_performance/{self.plannerName}.npy')
    #     with open(f"./datasave/planner_performance/{self.plannerName}_costgraph.pkl", "rb") as file:
    #         loadedList = pickle.load(file)

    #     print("Calculating Data ...")
    #     for i in range(plannerData.shape[1]):
    #         mv = plannerData[:,i]
    #         self.recordToPaper.append(f"{np.mean(mv)} +- {np.std(mv)}")

    #     print("Record this to paper")
    #     print(self.recordToPaper)

if __name__ == "__main__":
    plannerName = "starconnect_multi"
    trun = TrialRun(100, plannerName)
    trun.run()
    # trun.calculate()
    # fig, ax = plt.subplots()
    # trun.plot(ax)
    # plt.show()
