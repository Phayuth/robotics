import os
import sys
wd = os.path.abspath(os.getcwd())
sys.path.append(str(wd))

import time
import numpy as np
from test.project_main.workingcode.rrt_mydev_multi import RRTMyDevMultiGoal


class RRTMyDevMultiGoal2D(RRTMyDevMultiGoal):

    def __init__(self, xStart, xApp, xGoal, eta=0.3, subEta=0.05, maxIteration=2000, numDoF=2, envChoice="Planar", nearGoalRadius=0.3):
        super().__init__(xStart=xStart, xApp=xApp, xGoal=xGoal, eta=eta, subEta=subEta, maxIteration=maxIteration, numDoF=numDoF, envChoice=envChoice, nearGoalRadius=nearGoalRadius)

    def planning(self):
        timePlanningStart = time.perf_counter_ns()
        itera = self.rrt_mydev_multi()
        path1 = self.search_best_cost_singledirection_path(backFromNode=self.xAppList[0], treeVertexList=self.XSolnList[0], attachNode=self.xGoalList[0])
        path2 = self.search_best_cost_singledirection_path(backFromNode=self.xAppList[1], treeVertexList=self.XSolnList[1], attachNode=self.xGoalList[1])
        path3 = self.search_best_cost_singledirection_path(backFromNode=self.xAppList[2], treeVertexList=self.XSolnList[2], attachNode=self.xGoalList[2])
        timePlanningEnd = time.perf_counter_ns()

        # record performance
        self.perfMatrix["Planner Name"] = "Proposed + MultiGoal"
        self.perfMatrix["Parameters"]["eta"] = self.eta 
        self.perfMatrix["Parameters"]["subEta"] = self.subEta
        self.perfMatrix["Parameters"]["Max Iteration"] = self.maxIteration
        self.perfMatrix["Parameters"]["Rewire Radius"] = self.rewireRadius
        self.perfMatrix["Number of Node"] = len(self.treeVertexStart)
        self.perfMatrix["Total Planning Time"] = (timePlanningEnd-timePlanningStart) * 1e-9
        self.perfMatrix["KCD Time Spend"] = self.perfMatrix["KCD Time Spend"]* 1e-9
        self.perfMatrix["Planning Time Only"] = self.perfMatrix["Total Planning Time"] - self.perfMatrix["KCD Time Spend"]
        self.perfMatrix["Average KCD Time"] = self.perfMatrix["KCD Time Spend"] / self.perfMatrix["Number of Collision Check"]
        
        return path1, path2, path3

if __name__ == "__main__":
    np.random.seed(9)
    import matplotlib.pyplot as plt
    # import seaborn as sns
    # sns.set_theme()
    # sns.set_context("paper")
    from planner_util.coord_transform import circle_plt
    from util.general_util import write_dict_to_file

    xStart = np.array([0, 0]).reshape(2, 1)

    xApp = [np.array([np.pi/2-0.1, 0.2]).reshape(2, 1),
            np.array([1.45, -0.191]).reshape(2, 1),
            np.array([1.73, -0.160]).reshape(2, 1)]
    
    xGoal = [np.array([np.pi/2, 0]).reshape(2, 1), 
             np.array([np.pi/2, 0]).reshape(2, 1),
             np.array([np.pi/2, 0]).reshape(2, 1)]

    planner = RRTMyDevMultiGoal2D(xStart, xApp, xGoal, eta=0.1, maxIteration=2000)
    planner.robotEnv.robot.plot_arm(xStart, plt_basis=True)
    planner.robotEnv.robot.plot_arm(xGoal[0])
    planner.robotEnv.robot.plot_arm(xGoal[1])
    planner.robotEnv.robot.plot_arm(xGoal[2])
    planner.robotEnv.robot.plot_arm(xApp[0])
    planner.robotEnv.robot.plot_arm(xApp[1])
    planner.robotEnv.robot.plot_arm(xApp[2])
    for obs in planner.robotEnv.taskMapObs:
        obs.plot()
    plt.show()

    path1, path2, path3 = planner.planning()
    print(f"==>> path1: \n{path1}")
    print(f"==>> path2: \n{path2}")
    print(f"==>> path3: \n{path3}")
    print(planner.perfMatrix)
    # write_dict_to_file(planner.perfMatrix, "./planner_dev/result_2d/result_2d_proposed_2000.txt")
    
    fig, ax = plt.subplots()
    fig.set_size_inches(w=3.40067, h=3.40067)
    fig.tight_layout()
    # circle_plt(planner.xGoal1.x, planner.xGoal1.y, planner.distGoalToAppList[0])
    # circle_plt(planner.xGoal2.x, planner.xGoal2.y, planner.distGoalToAppList[1])
    # circle_plt(planner.xGoal3.x, planner.xGoal3.y, planner.distGoalToAppList[2])
    plt.xlim((-np.pi, np.pi))
    plt.ylim((-np.pi, np.pi))
    planner.plot_tree(path1, path2, path3, ax)
    plt.show()