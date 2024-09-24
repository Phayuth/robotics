import os
import sys

sys.path.append(str(os.path.abspath(os.getcwd())))

import time
import matplotlib.pyplot as plt
import numpy as np
import pickle
import pandas as pd
from spatial_geometry.spatial_transformation import RigidBodyTransformation as rbt

from datasave.joint_value.experiment_paper import URHarvesting
from datasave.joint_value.experiment_paper import ICRABarnMap
from spatial_geometry.utils import Utils

from simulator.sim_planar_rr import RobotArm2DSimulator
from simulator.sim_rectangle import TaskSpace2DSimulator
from simulator.sim_ur5e_api import UR5eArmCoppeliaSimAPI

from planner.sampling_based.rrt_plannerapi import RRTPlannerAPI


class TrialRun:

    def __init__(self, numTrial, plannerid, simulator, q, locgap) -> None:
        self.xStart = q.xStart
        self.xApp = q.xApp
        self.xGoal = q.xGoal
        self.numTrial = numTrial

        if simulator.__class__.__name__ == "RobotArm2DSimulator":
            self.configPlanner = {
                "planner": plannerid,
                "eta": 0.3,
                "subEta": 0.05,
                "maxIteration": 2000,
                "simulator": simulator,
                "nearGoalRadius": None,
                "rewireRadius": None,
                "endIterationID": 1,
                "printDebug": True,
                "localOptEnable": locgap,
            }

        elif simulator.__class__.__name__ == "UR5eArmCoppeliaSimAPI":
            self.configPlanner = {
                "planner": plannerid,
                "eta": 0.15,
                "subEta": 0.05,
                "maxIteration": 3000,
                "simulator": simulator,
                "nearGoalRadius": None,
                "rewireRadius": None,
                "endIterationID": 1,
                "printDebug": True,
                "localOptEnable": locgap
            }

        self.data = {
            "First Cost": [],
            "First Iteration": [],
            "Latest Cost": [],
            "Latest Iteration": [],
            "Run Time": [],
            "Plan Time": [],
            "Col.C Time": [],
            "Node Number": [],
            "Col.C Number": [],
        }

        self.costGraph = []

        if isinstance(self.xGoal, list):
            types = "Multi"
        else:
            types = "Single"

        self.namefile = f"env_{simulator.__class__.__name__}_types_{types}_planner_{plannerid}_withlocgap_{locgap}_"

    def run(self):
        for i in range(self.numTrial):
            np.random.seed(i + 1)
            print(f"Trail {i}-th")

            time.sleep(2)
            pm = RRTPlannerAPI.init_warp_q_to_pi(self.xStart, self.xApp, self.xGoal, self.configPlanner)
            patha = pm.begin_planner()

            cond = len(pm.planner.perfMatrix["Cost Graph"])  # check if planner has solution
            self.data["First Cost"].append(pm.planner.perfMatrix["Cost Graph"][0][1] if cond != 0 else None)
            self.data["First Iteration"].append(pm.planner.perfMatrix["Cost Graph"][0][0] if cond != 0 else None)
            self.data["Latest Cost"].append(pm.planner.perfMatrix["Cost Graph"][-1][1] if cond != 0 else None)
            self.data["Latest Iteration"].append(pm.planner.perfMatrix["Cost Graph"][-1][0] if cond != 0 else None)
            self.data["Run Time"].append(pm.planner.perfMatrix["Total Planning Time"])
            self.data["Plan Time"].append(pm.planner.perfMatrix["Planning Time Only"])
            self.data["Col.C Time"].append(pm.planner.perfMatrix["KCD Time Spend"])
            self.data["Node Number"].append(pm.planner.perfMatrix["Number of Node"])
            self.data["Col.C Number"].append(pm.planner.perfMatrix["Number of Collision Check"])

            self.costGraph.append(pm.planner.perfMatrix["Cost Graph"])

        print("Finished Trail Loop")

        # save data
        path = f"./datasave/new_paper/"

        # table
        df = pd.DataFrame(self.data)
        df.to_csv(path + self.namefile +"performance.csv", index=True)

        # graph
        with open(path + self.namefile +"costgraph.pkl", "wb") as file:
            pickle.dump(self.costGraph, file)


def case_1_multi():
    trailrun = 100

    simulator = RobotArm2DSimulator()

    class RobotArmPose:
        rc = 0.8
        g1 = rbt.conv_polar_to_cartesian(rc, 0.0, -3.0, 0.5)
        g2 = rbt.conv_polar_to_cartesian(rc, 0.5, -3.0, 0.5)  # intended collision
        g3 = rbt.conv_polar_to_cartesian(rc, -0.5, -3.0, 0.5)

        robot = simulator.robot
        q1 = robot.inverse_kinematic_geometry(np.array(g1).reshape(2, 1), elbow_option=1)
        q2 = robot.inverse_kinematic_geometry(np.array(g2).reshape(2, 1), elbow_option=1)
        q3 = robot.inverse_kinematic_geometry(np.array(g3).reshape(2, 1), elbow_option=1)

        q1i = robot.inverse_kinematic_geometry(np.array(g1).reshape(2, 1), elbow_option=0)
        q2i = robot.inverse_kinematic_geometry(np.array(g2).reshape(2, 1), elbow_option=0)
        q3i = robot.inverse_kinematic_geometry(np.array(g3).reshape(2, 1), elbow_option=0)

        q1i = Utils.wrap_to_pi(q1i)
        q2i = Utils.wrap_to_pi(q2i)
        q3i = Utils.wrap_to_pi(q3i)

        xStart = np.array([0, 0]).reshape(2, 1)
        xApp = [q1, q3, q1i, q2i, q3i]
        xGoal = [q1, q3, q1i, q2i, q3i]

    q = RobotArmPose()

    # case 1 rrt cnt multi
    plannerid = 12
    trun = TrialRun(trailrun, plannerid, simulator, q, locgap=False)
    trun.run()

    # case 1 rrt cnt na multi
    plannerid = 16
    trun = TrialRun(trailrun, plannerid, simulator, q, locgap=False)
    trun.run()

    # case 1 rrt star cnt multi
    plannerid = 13
    trun = TrialRun(trailrun, plannerid, simulator, q, locgap=False)
    trun.run()

    # case 1 rrt cnt locgap multi
    plannerid = 12
    trun = TrialRun(trailrun, plannerid, simulator, q, locgap=True)
    trun.run()

    # case 1 rrt cnt na locgap multi
    plannerid = 16
    trun = TrialRun(trailrun, plannerid, simulator, q, locgap=True)
    trun.run()

    # case 1 rrt star cnt locgap multi
    plannerid = 13
    trun = TrialRun(trailrun, plannerid, simulator, q, locgap=True)
    trun.run()



if __name__ == "__main__":

    # run
    case_1_multi()