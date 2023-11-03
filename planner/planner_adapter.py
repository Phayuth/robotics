import os
import sys

wd = os.path.abspath(os.getcwd())
sys.path.append(str(wd))

import time
import numpy as np
from planner.rrt_base import RRTBase
from planner.rrt_connect import RRTConnect
from planner.rrt_star import RRTStar
from planner.rrt_informed import RRTInformed
from planner.rrt_star_connect import RRTStarConnect
from planner.rrt_informed_connect import RRTInformedConnect
from planner.rrt_connect_ast_informed import RRTConnectAstInformed

from planner.rrt_base import RRTBaseMulti
from planner.rrt_connect import RRTConnectMulti
from planner.rrt_star import RRTStarMulti
from planner.rrt_informed import RRTInformedMulti
from planner.rrt_star_connect import RRTStarConnectMulti


class PlannerJointProcess:

    def wrap_to_pi(pose):
        return (pose + np.pi) % (2 * np.pi) - np.pi

    def get_trajectory(path, time, jointSpeedLimit):
        pass


class PlanAdapterCopSim:

    def __init__(self, xStart, xApp, xGoal, ID):

        eta = 0.15
        subEta = 0.05
        maxIteration = 3000
        numDoF = 6
        envChoice = "CopSim"
        nearGoalRadius = None #0.3
        rewireRadius = None
        endIterationID = 1
        print_debug = True
        localOptEnable = True

        # process joint
        xStart = PlannerJointProcess.wrap_to_pi(xStart)
        if isinstance(xApp, list):
            xApp = [PlannerJointProcess.wrap_to_pi(x) for x in xApp]
            xGoal = [PlannerJointProcess.wrap_to_pi(x) for x in xGoal]
        else:
            xApp = PlannerJointProcess.wrap_to_pi(xApp)
            xGoal = PlannerJointProcess.wrap_to_pi(xGoal)

        # Single
        if ID == 1:
            self.planner = RRTBase(xStart, xApp, xGoal, eta, subEta, maxIteration, numDoF, envChoice, nearGoalRadius, rewireRadius, endIterationID, print_debug)
        elif ID == 2:
            self.planner = RRTConnect(xStart, xApp, xGoal, eta, subEta, maxIteration, numDoF, envChoice, nearGoalRadius, rewireRadius, endIterationID, print_debug, localOptEnable)
        elif ID == 3:
            self.planner = RRTStar(xStart, xApp, xGoal, eta, subEta, maxIteration, numDoF, envChoice, nearGoalRadius, rewireRadius, endIterationID, print_debug, localOptEnable)
        elif ID == 4:
            self.planner = RRTInformed(xStart, xApp, xGoal, eta, subEta, maxIteration, numDoF, envChoice, nearGoalRadius, rewireRadius, endIterationID, print_debug)
        elif ID == 5:
            self.planner = RRTStarConnect(xStart, xApp, xGoal, eta, subEta, maxIteration, numDoF, envChoice, nearGoalRadius, rewireRadius, endIterationID, print_debug, localOptEnable)
        elif ID == 6:
            self.planner = RRTInformedConnect(xStart, xApp, xGoal, eta, subEta, maxIteration, numDoF, envChoice, nearGoalRadius, rewireRadius, endIterationID, print_debug)
        elif ID == 7:
            self.planner = RRTConnectAstInformed(xStart, xApp, xGoal, eta, subEta, maxIteration, numDoF, envChoice, nearGoalRadius, rewireRadius, endIterationID, print_debug)

        # Multi
        elif ID == 8:
            self.planner = RRTBaseMulti(xStart, xApp, xGoal, eta, subEta, maxIteration, numDoF, envChoice, nearGoalRadius, rewireRadius, endIterationID, print_debug)
        elif ID == 9:
            self.planner = RRTConnectMulti(xStart, xApp, xGoal, eta, subEta, maxIteration, numDoF, envChoice, nearGoalRadius, rewireRadius, endIterationID, print_debug, localOptEnable)
        elif ID == 10:
            self.planner = RRTStarMulti(xStart, xApp, xGoal, eta, subEta, maxIteration, numDoF, envChoice, nearGoalRadius, rewireRadius, endIterationID, print_debug)
        elif ID == 11:
            self.planner = RRTInformedMulti(xStart, xApp, xGoal, eta, subEta, maxIteration, numDoF, envChoice, nearGoalRadius, rewireRadius, endIterationID, print_debug)
        elif ID == 12:
            self.planner = RRTStarConnectMulti(xStart, xApp, xGoal, eta, subEta, maxIteration, numDoF, envChoice, nearGoalRadius, rewireRadius, endIterationID, print_debug, localOptEnable)

    def planning(self):
        self.planner.robotEnv.start_sim()
        timePlanningStart = time.perf_counter_ns()
        self.planner.start()
        path = self.planner.get_path()
        timePlanningEnd = time.perf_counter_ns()
        self.planner.update_perf(timePlanningStart, timePlanningEnd)
        self.planner.robotEnv.stop_sim()
        return path


class PlanAdapterPlanar2D:

    def __init__(self, xStart, xApp, xGoal, ID):

        eta = 0.3
        subEta = 0.05
        maxIteration = 2000
        numDoF = 2
        envChoice = "Planar"
        nearGoalRadiusSingleTree = 0.3
        nearGoalRadiusDualTree = None
        rewireRadius = None
        endIterationID = 1
        print_debug = False
        localOptEnable = True
        
        # Single
        if ID == 1:
            self.planner = RRTBase(xStart, xApp, xGoal, eta, subEta, maxIteration, numDoF, envChoice, nearGoalRadiusSingleTree, rewireRadius, endIterationID, print_debug)
        elif ID == 2:
            self.planner = RRTConnect(xStart, xApp, xGoal, eta, subEta, maxIteration, numDoF, envChoice, nearGoalRadiusDualTree, rewireRadius, endIterationID, print_debug, localOptEnable)
        elif ID == 3:
            self.planner = RRTStar(xStart, xApp, xGoal, eta, subEta, maxIteration, numDoF, envChoice, nearGoalRadiusSingleTree, rewireRadius, endIterationID, print_debug, localOptEnable)
        elif ID == 4:
            self.planner = RRTInformed(xStart, xApp, xGoal, eta, subEta, maxIteration, numDoF, envChoice, nearGoalRadiusSingleTree, rewireRadius, endIterationID, print_debug)
        elif ID == 5:
            self.planner = RRTStarConnect(xStart, xApp, xGoal, eta, subEta, maxIteration, numDoF, envChoice, nearGoalRadiusSingleTree, rewireRadius, endIterationID, print_debug, localOptEnable)
        elif ID == 6:
            self.planner = RRTInformedConnect(xStart, xApp, xGoal, eta, subEta, maxIteration, numDoF, envChoice, nearGoalRadiusSingleTree, rewireRadius, endIterationID, print_debug)
        elif ID == 7:
            self.planner = RRTConnectAstInformed(xStart, xApp, xGoal, eta, subEta, maxIteration, numDoF, envChoice, nearGoalRadiusDualTree, rewireRadius, endIterationID, print_debug)

        # Multi
        elif ID == 8:
            self.planner = RRTBaseMulti(xStart, xApp, xGoal, eta, subEta, maxIteration, numDoF, envChoice, nearGoalRadiusSingleTree, rewireRadius, endIterationID, print_debug)
        elif ID == 9:
            self.planner = RRTConnectMulti(xStart, xApp, xGoal, eta, subEta, maxIteration, numDoF, envChoice, nearGoalRadiusDualTree, rewireRadius, endIterationID, print_debug, localOptEnable)
        elif ID == 10:
            self.planner = RRTStarMulti(xStart, xApp, xGoal, eta, subEta, maxIteration, numDoF, envChoice, nearGoalRadiusSingleTree, rewireRadius, endIterationID, print_debug)
        elif ID == 11:
            self.planner = RRTInformedMulti(xStart, xApp, xGoal, eta, subEta, maxIteration, numDoF, envChoice, nearGoalRadiusSingleTree, rewireRadius, endIterationID, print_debug)
        elif ID == 12:
            self.planner = RRTStarConnectMulti(xStart, xApp, xGoal, eta, subEta, maxIteration, numDoF, envChoice, nearGoalRadiusDualTree, rewireRadius, endIterationID, print_debug, localOptEnable)

    def planning(self):
        timePlanningStart = time.perf_counter_ns()
        self.planner.start()
        path = self.planner.get_path()
        timePlanningEnd = time.perf_counter_ns()
        self.planner.update_perf(timePlanningStart, timePlanningEnd)
        return path


if __name__ == "__main__":
    np.random.seed(0)
    import matplotlib.pyplot as plt
    # from datasave.joint_value.pre_record_value import  newThetaInit, newThetaApp, newThetaGoal





    # ----------------------------------------------------------------------------------------------------------------
    # Single
    # xStart = newThetaInit
    # xGoal = newThetaGoal
    # xApp = newThetaApp

    # Multi
    # xStart = newThetaInit
    # xGoal = [newThetaGoal, newThetaGoal, newThetaGoal]
    # xApp = [newThetaApp, newThetaApp, newThetaApp]

    # pa = PlanAdapterCopSim(xStart, xApp, xGoal, 12)
    # path = pa.planning()
    # time.sleep(3)
    # pa.planner.robotEnv.play_back_path(path)







    # ----------------------------------------------------------------------------------------------------------------
    # Single Arm
    # xStart = np.array([0, 0]).reshape(2, 1)
    # xGoal = np.array([np.pi / 2, 0]).reshape(2, 1)
    # xApp = np.array([np.pi / 2 - 0.1, 0.2]).reshape(2, 1)

    # Multi
    # xStart = np.array([0, 0]).reshape(2, 1)
    # xApp = [np.array([np.pi / 2 - 0.1, 0.2]).reshape(2, 1), np.array([1.45, -0.191]).reshape(2, 1), np.array([1.73, -0.160]).reshape(2, 1)]
    # xGoal = [np.array([np.pi / 2, 0]).reshape(2, 1), np.array([np.pi / 2, 0]).reshape(2, 1), np.array([np.pi / 2, 0]).reshape(2, 1)]

    # Single Task
    # xStart = np.array([[-2.70], [2.20]])
    # xGoal = np.array([[2.30], [-2.30]])
    # xApp = np.array([[2.30], [-1.85]])

    # Multi Task
    xStart = np.array([[-2.70], [2.20]])
    xApp = [np.array([[2.30], [-1.85]]), 
            np.array([[1.81], [-2.06]]), 
            np.array([[1.80], [-2.60]])]
    xGoal = [np.array([[2.30], [-2.30]]), 
             np.array([[2.30], [-2.30]]), 
             np.array([[2.30], [-2.30]])]

    pa = PlanAdapterPlanar2D(xStart, xApp, xGoal, 9)

    # pa.planner.robotEnv.robot.plot_arm(xStart, plt_basis=True)
    # pa.planner.robotEnv.robot.plot_arm(xGoal)
    # pa.planner.robotEnv.robot.plot_arm(xApp)
    # for obs in pa.planner.robotEnv.taskMapObs:
    #     obs.plot()
    # plt.show()

    path = pa.planning()
    print(pa.planner.perfMatrix)
    fig, ax = plt.subplots()
    fig.set_size_inches(w=3.40067, h=3.40067)
    fig.tight_layout()
    plt.xlim((-np.pi, np.pi))
    plt.ylim((-np.pi, np.pi))
    pa.planner.plot_tree(path, ax)
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




    # ----------------------------------------------------------------------------------------------------------------
    fig, ax = plt.subplots()
    pa.planner.plot_performance(ax)
    plt.show()