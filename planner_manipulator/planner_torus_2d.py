import os
import sys

wd = os.path.abspath(os.getcwd())
sys.path.append(str(wd))

import time
from joint_process import JointProcess


class TorusPlannerManipulator:

    def __init__(self, xStart, xApp, xGoal, config):
        self.planner = config["planner"](xStart, xApp, xGoal, config)

    def planning(self):
        timePlanningStart = time.perf_counter_ns()
        self.planner.start()
        path = self.planner.get_path()
        timePlanningEnd = time.perf_counter_ns()
        self.planner.update_perf(timePlanningStart, timePlanningEnd)
        return path

if __name__ == "__main__":
    import numpy as np
    np.random.seed(0)
    import matplotlib.pyplot as plt

    # planning algorithm
    from planner.rrt_base import RRTBase, RRTBaseMulti
    from planner.rrt_connect import RRTConnect, RRTConnectMulti
    from planner.rrt_star import RRTStar, RRTStarMulti
    from planner.rrt_informed import RRTInformed, RRTInformedMulti
    from planner.rrt_star_connect import RRTStarConnect, RRTStarConnectMulti
    from planner.rrt_informed_connect import RRTInformedConnect
    from planner.rrt_connect_ast_informed import RRTConnectAstInformed
    from planner.rrt_star_quick import RRTStarQuick, RRTStarQuickMulti
    from planner.rrt_star_connect_quick import RRTStarConnectQuick, RRTStarConnectQuickMulti

    # environment
    from map.sim_2d_env import RobotArm2DEnvironment

    # plotter
    from planner.rrt_plotter import RRTPlotter


    plannarConfigSingleTree = {
        "planner": RRTStarMulti,
        "eta": 0.3,
        "subEta": 0.05,
        "maxIteration": 10000,
        "robotEnvClass": RobotArm2DEnvironment,
        "nearGoalRadius": 0.2,
        "rewireRadius": None,
        "endIterationID": 1,
        "print_debug": True,
        "localOptEnable": False
    }

    plannarConfigDualTreea = {
        "planner": RRTStarConnectQuickMulti,
        "eta": 0.3,
        "subEta": 0.05,
        "maxIteration": 10000,
        "robotEnvClass": RobotArm2DEnvironment,
        "nearGoalRadius": None,
        "rewireRadius": None,
        "endIterationID": 1,
        "print_debug": True,
        "localOptEnable": True
    }

    xStart = np.array([0.0, -3.0]).reshape(2,1)

    xApp = [np.array([1.47079633, 0.2]).reshape(2,1),
             np.array([1.47079633, -6.08318531]).reshape(2,1),
             np.array([-4.81238898, 0.2]).reshape(2,1),
             np.array([-4.81238898, -6.08318531]).reshape(2,1)]

    xGoal = [np.array([1.47079633, 0.2]).reshape(2,1),
             np.array([1.47079633, -6.08318531]).reshape(2,1),
             np.array([-4.81238898, 0.2]).reshape(2,1),
             np.array([-4.81238898, -6.08318531]).reshape(2,1)]

    pa = TorusPlannerManipulator(xStart, xApp, xGoal, plannarConfigDualTreea)

    patha = pa.planning()
    fig, ax = plt.subplots()
    fig.set_size_inches(w=3.40067, h=3.40067)
    fig.tight_layout()
    plt.xlim((-2*np.pi, 2*np.pi))
    plt.ylim((-2*np.pi, 2*np.pi))
    # RRTPlotter.plot_2d_config_single_tree(pa.planner, patha, ax)
    RRTPlotter.plot_2d_config_dual_tree(pa.planner, patha, ax)

    plt.show()

    fig, ax = plt.subplots()
    RRTPlotter.plot_performance(pa.planner.perfMatrix, ax)
    plt.show()