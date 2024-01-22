import os
import sys

wd = os.path.abspath(os.getcwd())
sys.path.append(str(wd))

import time
import matplotlib.pyplot as plt
import numpy as np
import pickle

from datasave.joint_value.experiment_paper import ICRABarnMap
from simulator.sim_planar_rr import RobotArm2DSimulator
from planner.planner_manipulator import PlannerManipulator


class TrialRun:

    def __init__(self, numTrial) -> None:
        # q = ICRABarnMap.PoseSingle()
        q = ICRABarnMap.PoseMulti3()
        self.xStart = q.xStart
        self.xApp = q.xApp
        self.xGoal = q.xGoal

        self.simu = RobotArm2DSimulator()
        self.configPlanner = {
            "planner": 13,
            "eta": 0.3,
            "subEta": 0.05,
            "maxIteration": 2000,
            "simulator": self.simu,
            "nearGoalRadius": None,
            "rewireRadius": None,
            "endIterationID": 1,
            "printDebug": True,
            "localOptEnable": True
        }

        # self.plannerName = plannerName
        self.perf = ['Number of Node',
                     'First Path Cost',
                     'First Path Iteration',
                     'Final Path Cost',
                     'Final Path Iteration'
                     'Running Time (sec)',
                     'Planner Time (sec)',
                     'Col. Check Time (sec)',
                     'Number of Col. Check',
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
            pm = PlannerManipulator(self.xStart, self.xApp, self.xGoal, self.configPlanner)
            patha = pm.planning()

            data = [pm.planner.perfMatrix["Number of Node"],
                    pm.planner.perfMatrix["Cost Graph"][0][1],
                    pm.planner.perfMatrix["Cost Graph"][0][0],
                    pm.planner.perfMatrix["Cost Graph"][-1][1],
                    pm.planner.perfMatrix["Cost Graph"][-1][0],
                    pm.planner.perfMatrix["Total Planning Time"],
                    pm.planner.perfMatrix["Planning Time Only"],
                    pm.planner.perfMatrix["KCD Time Spend"],
                    pm.planner.perfMatrix["Number of Collision Check"],
                    pm.planner.perfMatrix["Average KCD Time"]]

            self.plannerData.append(data)
            self.costGraph.append(pm.planner.perfMatrix["Cost Graph"])

        print("Finished Trail Loop")

        plannerData = np.array(self.plannerData)
        print(f"==>> plannerData: {plannerData}")

        # save data
        # np.save(f'./datasave/planner_performance/{self.plannerName}.npy', plannerData)
        # fileName = f"./datasave/planner_performance/{self.plannerName}_costgraph.pkl"
        # with open(fileName, "wb") as file:
        #     pickle.dump(self.costGraph, file)

if __name__ == "__main__":
    trun = TrialRun(3)
    trun.run()