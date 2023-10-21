import os
import sys
wd = os.path.abspath(os.getcwd())
sys.path.append(str(wd))

import time
import numpy as np
from test.project_main.workingcode.rrt_mydev_multi import RRTMyDevMultiGoal


class RRTMyDevMultiGoalCopSim6D(RRTMyDevMultiGoal):

    def __init__(self, xStart, xApp, xGoal, eta=0.3, subEta=0.05, maxIteration=2000, numDoF=6, envChoice="CopSim", nearGoalRadius=0.3):
        super().__init__(xStart=xStart, xApp=xApp, xGoal=xGoal, eta=eta, subEta=subEta, maxIteration=maxIteration, numDoF=numDoF, envChoice=envChoice, nearGoalRadius=nearGoalRadius)

    def planning(self):
        self.robotEnv.start_sim()

        timePlanningStart = time.perf_counter_ns()
        itera = self.rrt_mydev_multi()
        path1 = self.search_best_cost_singledirection_path(backFromNode=self.xAppList[0], treeVertexList=self.XSolnList[0], attachNode=self.xGoalList[0])
        path2 = self.search_best_cost_singledirection_path(backFromNode=self.xAppList[1], treeVertexList=self.XSolnList[1], attachNode=self.xGoalList[1])
        path3 = self.search_best_cost_singledirection_path(backFromNode=self.xAppList[2], treeVertexList=self.XSolnList[2], attachNode=self.xGoalList[2])
        timePlanningEnd = time.perf_counter_ns()

        # record performance
        self.perfMatrix["Planner Name"] = "Proposed"
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
        return path1, path2, path3


if __name__ == "__main__":
    np.random.seed(9)
    from datasave.joint_value.pre_record_value import wrap_to_pi, newThetaInit, newThetaApp, newThetaGoal
    from util.general_util import write_dict_to_file

    # Define pose
    xStart = newThetaInit
    xApp = [wrap_to_pi(newThetaApp), wrap_to_pi(newThetaApp), wrap_to_pi(newThetaApp)]
    xGoal = [wrap_to_pi(newThetaGoal), wrap_to_pi(newThetaGoal), wrap_to_pi(newThetaGoal)]

    planner = RRTMyDevMultiGoalCopSim6D(xStart, xApp, xGoal, eta=0.3, maxIteration=5000)
    path1, path2, path3 = planner.planning()
    print(f"==>> path1: \n{path1}")
    print(f"==>> path2: \n{path2}")
    print(f"==>> path3: \n{path3}")
    print(planner.perfMatrix)
    # write_dict_to_file(planner.perfMatrix, "./planner_dev/result_6d/result_6d_proposed.txt")

    time.sleep(3)

    # # play back
    # planner.robotEnv.start_sim()
    # planner.robotEnv.set_goal_joint_value(thetaGoal)
    # planner.robotEnv.set_aux_joint_value(thetaApp)
    # planner.robotEnv.set_start_joint_value(thetaInit)
    # for i in range(len(path)):
    #     planner.robotEnv.set_joint_value(path[i])
    #     time.sleep(2)
    #     # triggers next simulation step
    #     # client.step()

    # # stop simulation
    # planner.robotEnv.stop_sim()