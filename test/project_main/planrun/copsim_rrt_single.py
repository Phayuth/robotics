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


class RRTBaseCopSim(RRTBase):

    def __init__(self, 
                 xStart, 
                 xApp, 
                 xGoal, 
                 eta=0.3, 
                 subEta=0.05, 
                 maxIteration=2000, 
                 numDoF=6, 
                 envChoice="CopSim", 
                 nearGoalRadius=0.3, 
                 rewireRadius=None, 
                 terminationConditionID=1, 
                 print_debug=False):
        super().__init__(xStart=xStart,
                         xApp=xApp,
                         xGoal=xGoal,
                         eta=eta,
                         subEta=subEta,
                         maxIteration=maxIteration,
                         numDoF=numDoF,
                         envChoice=envChoice,
                         nearGoalRadius=nearGoalRadius,
                         rewireRadius=rewireRadius,
                         terminationConditionID=terminationConditionID,
                         print_debug=print_debug)

    def planning(self):
        self.robotEnv.start_sim()
        self.robotEnv.set_start_joint_value(self.xStart)
        self.robotEnv.set_aux_joint_value(self.xApp)
        self.robotEnv.set_goal_joint_value(self.xGoal)
        timePlanningStart = time.perf_counter_ns()
        self.start()
        path = self.get_path()
        timePlanningEnd = time.perf_counter_ns()
        self.update_perf(timePlanningStart, timePlanningEnd)
        self.robotEnv.stop_sim()
        return path


class RRTConnectCopSim(RRTConnect):

    def __init__(self, 
                 xStart, 
                 xApp, 
                 xGoal, 
                 eta=0.3, 
                 subEta=0.05, 
                 maxIteration=2000, 
                 numDoF=6, 
                 envChoice="CopSim", 
                 nearGoalRadius=None, 
                 rewireRadius=None, 
                 terminationConditionID=1, 
                 print_debug=False,
                 localOptEnable=False):
        super().__init__(xStart=xStart,
                         xApp=xApp,
                         xGoal=xGoal,
                         eta=eta,
                         subEta=subEta,
                         maxIteration=maxIteration,
                         numDoF=numDoF,
                         envChoice=envChoice,
                         nearGoalRadius=nearGoalRadius,
                         rewireRadius=rewireRadius,
                         terminationConditionID=terminationConditionID,
                         print_debug=print_debug,
                         localOptEnable=localOptEnable)

    def planning(self):
        self.robotEnv.start_sim()
        self.robotEnv.set_start_joint_value(self.xStart)
        self.robotEnv.set_aux_joint_value(self.xApp)
        self.robotEnv.set_goal_joint_value(self.xGoal)
        timePlanningStart = time.perf_counter_ns()
        self.start()
        path = self.get_path()
        timePlanningEnd = time.perf_counter_ns()
        self.update_perf(timePlanningStart, timePlanningEnd)
        self.robotEnv.stop_sim()
        return path


class RRTStarCopSim(RRTStar):

    def __init__(self, 
                 xStart, 
                 xApp, 
                 xGoal, 
                 eta=0.3, 
                 subEta=0.05, 
                 maxIteration=2000, 
                 numDoF=6, 
                 envChoice="CopSim", 
                 nearGoalRadius=0.3, 
                 rewireRadius=None, 
                 terminationConditionID=1, 
                 print_debug=False,
                 localOptEnable=False):
        super().__init__(xStart=xStart,
                         xApp=xApp,
                         xGoal=xGoal,
                         eta=eta,
                         subEta=subEta,
                         maxIteration=maxIteration,
                         numDoF=numDoF,
                         envChoice=envChoice,
                         nearGoalRadius=nearGoalRadius,
                         rewireRadius=rewireRadius,
                         terminationConditionID=terminationConditionID,
                         print_debug=print_debug,
                         localOptEnable=localOptEnable)

    def planning(self):
        self.robotEnv.start_sim()
        self.robotEnv.set_start_joint_value(self.xStart)
        self.robotEnv.set_aux_joint_value(self.xApp)
        self.robotEnv.set_goal_joint_value(self.xGoal)
        timePlanningStart = time.perf_counter_ns()
        self.start()
        path = self.get_path()
        timePlanningEnd = time.perf_counter_ns()
        self.update_perf(timePlanningStart, timePlanningEnd)
        self.robotEnv.stop_sim()
        return path


class RRTInformedCopSim(RRTInformed):

    def __init__(self, xStart, xApp, xGoal, eta=0.3, subEta=0.05, maxIteration=2000, numDoF=6, envChoice="CopSim", nearGoalRadius=0.3, rewireRadius=None, terminationConditionID=1, print_debug=False):
        super().__init__(xStart=xStart,
                         xApp=xApp,
                         xGoal=xGoal,
                         eta=eta,
                         subEta=subEta,
                         maxIteration=maxIteration,
                         numDoF=numDoF,
                         envChoice=envChoice,
                         nearGoalRadius=nearGoalRadius,
                         rewireRadius=rewireRadius,
                         terminationConditionID=terminationConditionID,
                         print_debug=print_debug)

    def planning(self):
        self.robotEnv.start_sim()
        self.robotEnv.set_start_joint_value(self.xStart)
        self.robotEnv.set_aux_joint_value(self.xApp)
        self.robotEnv.set_goal_joint_value(self.xGoal)
        timePlanningStart = time.perf_counter_ns()
        self.start()
        path = self.get_path()
        timePlanningEnd = time.perf_counter_ns()
        self.update_perf(timePlanningStart, timePlanningEnd)
        self.robotEnv.stop_sim()
        return path


class RRTStarConnectCopSim(RRTStarConnect):

    def __init__(self, 
                 xStart,
                 xApp, 
                 xGoal, 
                 eta=0.3, 
                 subEta=0.05, 
                 maxIteration=2000, 
                 numDoF=6, 
                 envChoice="CopSim", 
                 nearGoalRadius=None, 
                 rewireRadius=None, 
                 terminationConditionID=1, 
                 print_debug=False,
                 localOptEnable=False):
        super().__init__(xStart=xStart,
                         xApp=xApp,
                         xGoal=xGoal,
                         eta=eta,
                         subEta=subEta,
                         maxIteration=maxIteration,
                         numDoF=numDoF,
                         envChoice=envChoice,
                         nearGoalRadius=nearGoalRadius,
                         rewireRadius=rewireRadius,
                         terminationConditionID=terminationConditionID,
                         print_debug=print_debug,
                         localOptEnable=localOptEnable)

    def planning(self):
        self.robotEnv.start_sim()
        self.robotEnv.set_start_joint_value(self.xStart)
        self.robotEnv.set_aux_joint_value(self.xApp)
        self.robotEnv.set_goal_joint_value(self.xGoal)
        timePlanningStart = time.perf_counter_ns()
        self.start()
        path = self.get_path()
        timePlanningEnd = time.perf_counter_ns()
        self.update_perf(timePlanningStart, timePlanningEnd)
        self.robotEnv.stop_sim()
        return path


class RRTInformedConnectCopSim(RRTInformedConnect):

    def __init__(self, 
                 xStart, 
                 xApp, 
                 xGoal, 
                 eta=0.3, 
                 subEta=0.05, 
                 maxIteration=2000, 
                 numDoF=6, 
                 envChoice="CopSim", 
                 nearGoalRadius=None, 
                 rewireRadius=None, 
                 terminationConditionID=1, 
                 print_debug=False):
        super().__init__(xStart=xStart,
                         xApp=xApp,
                         xGoal=xGoal,
                         eta=eta,
                         subEta=subEta,
                         maxIteration=maxIteration,
                         numDoF=numDoF,
                         envChoice=envChoice,
                         nearGoalRadius=nearGoalRadius,
                         rewireRadius=rewireRadius,
                         terminationConditionID=terminationConditionID,
                         print_debug=print_debug)

    def planning(self):
        self.robotEnv.start_sim()
        self.robotEnv.set_start_joint_value(self.xStart)
        self.robotEnv.set_aux_joint_value(self.xApp)
        self.robotEnv.set_goal_joint_value(self.xGoal)
        timePlanningStart = time.perf_counter_ns()
        self.start()
        path = self.get_path()
        timePlanningEnd = time.perf_counter_ns()
        self.update_perf(timePlanningStart, timePlanningEnd)
        self.robotEnv.stop_sim()
        return path


class RRTConnectAstInformedCopSim(RRTConnectAstInformed):

    def __init__(self, 
                 xStart, 
                 xApp, 
                 xGoal, 
                 eta=0.3, 
                 subEta=0.05, 
                 maxIteration=2000, 
                 numDoF=6, 
                 envChoice="CopSim", 
                 nearGoalRadius=0.3, 
                 rewireRadius=None, 
                 terminationConditionID=1, 
                 print_debug=False):
        super().__init__(xStart=xStart,
                         xApp=xApp,
                         xGoal=xGoal,
                         eta=eta,
                         subEta=subEta,
                         maxIteration=maxIteration,
                         numDoF=numDoF,
                         envChoice=envChoice,
                         nearGoalRadius=nearGoalRadius,
                         rewireRadius=rewireRadius,
                         terminationConditionID=terminationConditionID,
                         print_debug=print_debug)

    def planning(self):
        self.robotEnv.start_sim()
        self.robotEnv.set_start_joint_value(self.xStart)
        self.robotEnv.set_aux_joint_value(self.xApp)
        self.robotEnv.set_goal_joint_value(self.xGoal)
        timePlanningStart = time.perf_counter_ns()
        self.start()
        path = self.get_path()
        timePlanningEnd = time.perf_counter_ns()
        self.update_perf(timePlanningStart, timePlanningEnd)
        self.robotEnv.stop_sim()
        return path


if __name__ == "__main__":
    np.random.seed(0)
    import matplotlib.pyplot as plt
    from datasave.joint_value.pre_record_value import wrap_to_pi, newThetaInit, newThetaApp, newThetaGoal, new3dofThetaInit, new3dofThetaApp, new3dofThetaGoal
    from util.general_util import write_dict_to_file

    # 6DoF
    numDoF = 6
    thetaInit = newThetaInit
    thetaGoal = wrap_to_pi(newThetaGoal)
    thetaApp = wrap_to_pi(newThetaApp)

    # 3DoF
    # numDoF = 3
    # thetaInit = new3dofThetaInit
    # thetaGoal = wrap_to_pi(new3dofThetaGoal)
    # thetaApp = wrap_to_pi(new3dofThetaApp)

    maxIteration = 3000
    eta = 0.15
    # planner = RRTBaseCopSim(thetaInit, thetaApp, thetaGoal, eta=eta, maxIteration=maxIteration, numDoF=numDoF)
    # planner = RRTConnectCopSim(thetaInit, thetaApp, thetaGoal, eta=eta, maxIteration=maxIteration, numDoF=numDoF, localOptEnable=True)
    # planner = RRTConnectCopSim(thetaInit, thetaApp, thetaGoal, eta=eta, maxIteration=maxIteration, numDoF=numDoF, localOptEnable=False)
    # planner = RRTStarCopSim(thetaInit, thetaApp, thetaGoal, eta=eta, maxIteration=maxIteration, numDoF=numDoF, localOptEnable=True)
    # planner = RRTStarCopSim(thetaInit, thetaApp, thetaGoal, eta=eta, maxIteration=maxIteration, numDoF=numDoF, localOptEnable=False)
    # planner = RRTInformedCopSim(thetaInit, thetaApp, thetaGoal, eta=eta, maxIteration=maxIteration, numDoF=numDoF)
    planner = RRTStarConnectCopSim(thetaInit, thetaApp, thetaGoal, eta=eta, maxIteration=maxIteration, numDoF=numDoF, localOptEnable=True)
    # planner = RRTStarConnectCopSim(thetaInit, thetaApp, thetaGoal, eta=eta, maxIteration=maxIteration, numDoF=numDoF, localOptEnable=False)
    # planner = RRTInformedConnectCopSim(thetaInit, thetaApp, thetaGoal, eta=eta, maxIteration=maxIteration, numDoF=numDoF)
    # planner = RRTConnectAstInformedCopSim(thetaInit, thetaApp, thetaGoal, eta=eta, maxIteration=maxIteration, numDoF=numDoF)

    path = planner.planning()
    print(planner.perfMatrix)
    # write_dict_to_file(planner.perfMatrix, "./planner_dev/result_6d/result_6d_rrtstar.txt")
    print(f"==>> path: \n{path}")

    time.sleep(3)

    # play back
    planner.robotEnv.start_sim()
    planner.robotEnv.set_start_joint_value(thetaInit)
    planner.robotEnv.set_goal_joint_value(thetaGoal)
    planner.robotEnv.set_aux_joint_value(thetaApp)
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