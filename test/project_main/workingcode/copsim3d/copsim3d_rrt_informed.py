""" 
Devplanner with RRT Informed with reject goal sampling

"""
import os
import sys
wd = os.path.abspath(os.getcwd())
sys.path.append(str(wd))

import time
import numpy as np
from copsim3d_rrt_component import Node, CopSim3DRRTComponent


class RRTInformedDev(CopSim3DRRTComponent):
    def __init__(self, xStartFull, xAppFull, xGoalFull, eta=0.3, maxIteration=1000) -> None:
        super().__init__()
        # start, aux, goal node
        self.xStart = Node(xStartFull[0, 0], xStartFull[1, 0], xStartFull[2, 0])
        self.xGoal = Node(xGoalFull[0, 0], xGoalFull[1, 0], xGoalFull[2, 0])
        self.xApp = Node(xAppFull[0, 0], xAppFull[1, 0], xAppFull[2, 0])

        self.joint4Fixed = xAppFull[3, 0]
        self.joint5Fixed = xAppFull[4, 0]
        self.joint6Fixed = xAppFull[5, 0]

        self.eta = eta
        self.subEta = 0.05
        self.nearGoalRadius = 0.5
        self.maxIteration = maxIteration
        self.treeVertex = [self.xStart]
        self.distGoalToApp = self.distance_between_config(self.xGoal, self.xApp)
        self.rewireRadius = 0.5
        self.XSoln = []

    def planning(self):
        self.copHandle.start_sim()

        timePlanningStart = time.perf_counter_ns()
        itera = self.planner_rrt_informed()
        path = self.search_best_cost_singledirection_path(backFromNode=self.xApp, treeVertexList=self.XSoln, attachNode=self.xGoal)
        timePlanningEnd = time.perf_counter_ns()

        # record performance
        self.perfMatrix["Planner Name"] = "Informed RRT Star NewMod"
        self.perfMatrix["Parameters"]["eta"] = self.eta 
        self.perfMatrix["Parameters"]["subEta"] = self.subEta
        self.perfMatrix["Parameters"]["Max Iteration"] = self.maxIteration
        self.perfMatrix["Parameters"]["Rewire Radius"] = self.rewireRadius
        self.perfMatrix["Number of Node"] = len(self.treeVertex)
        self.perfMatrix["Total Planning Time"] = (timePlanningEnd-timePlanningStart) * 1e-9
        self.perfMatrix["KCD Time Spend"] = self.perfMatrix["KCD Time Spend"]* 1e-9
        self.perfMatrix["Planning Time Only"] = self.perfMatrix["Total Planning Time"] - self.perfMatrix["KCD Time Spend"]
        self.perfMatrix["Average KCD Time"] = self.perfMatrix["KCD Time Spend"] / self.perfMatrix["Number of Collision Check"]

        self.copHandle.stop_sim()
        return path

    def planner_rrt_informed(self):
        for itera in range(self.maxIteration):
            print(itera)
            if len(self.XSoln) == 0:
                cBest = np.inf
                cBestPrevious = np.inf
            else:
                xSolnCost = [xSoln.parent.cost + self.cost_line(xSoln.parent, xSoln) + self.cost_line(xSoln, self.xApp) for xSoln in self.XSoln]
                print(f"==>> xSolnCost: \n{xSolnCost}")
                cBest = min(xSolnCost)
                if cBest < cBestPrevious : # this have nothing to do with planning itself, just for record performance data only
                    self.perfMatrix["Cost Graph"].append((itera, cBest))
                    cBestPrevious = cBest

            xRand = self.informed_sampling(self.xStart, self.xApp, cBest, biasToNode=None) #self.xApp) # this has bias careful
            xNearest = self.nearest_node(self.treeVertex, xRand)
            xNew = self.steer(xNearest, xRand, self.eta)
            if self.is_collision_and_in_goal_region(xNearest, xNew, self.xGoal, self.distGoalToApp):
                continue
            xNew.parent = xNearest
            xNew.cost = xNew.parent.cost + self.cost_line(xNew, xNew.parent)
            XNear = self.near(self.treeVertex, xNew, self.rewireRadius)
            xMin = xNew.parent
            cMin = xMin.cost + self.cost_line(xMin, xNew)
            for xNear in XNear:
                if self.is_connect_config_in_collision(xNear, xNew):
                    continue

                cNew = xNear.cost + self.cost_line(xNear, xNew)
                if cNew < cMin:
                    xMin = xNear
                    cMin = cNew

            xNew.parent = xMin
            xNew.cost = cMin
            self.treeVertex.append(xNew)

            for xNear in XNear:
                if self.is_connect_config_in_collision(xNear, xNew):
                    continue
                cNear = xNear.cost
                cNew = xNew.cost + self.cost_line(xNew, xNear)
                if cNew < cNear:
                    xNear.parent = xNew
                    xNear.cost = xNew.cost + self.cost_line(xNew, xNear)

            # in approach region
            if self.is_config_in_region_of_config(xNew, self.xApp, radius=self.nearGoalRadius):
                self.XSoln.append(xNew)
                print("Added Near Region")

        return itera

    def informed_sampling(self, xStart, xGoal, cMax, biasToNode=None):
        if cMax < np.inf:
            cMin = self.distance_between_config(xStart, xGoal)
            print(cMax, cMin)
            xCenter = np.array([(xStart.x + xGoal.x) / 2,
                                (xStart.y + xGoal.y) / 2,
                                (xStart.z + xGoal.z) / 2]).reshape(3, 1)

            L, C = self.rotation_to_world(xStart, xGoal, cMax, cMin)

            while True:
                xBall = self.unit_ball_sampling()
                xRand = (C@L@xBall) + xCenter
                xRand = Node(xRand[0, 0], xRand[1, 0], xRand[2, 0])
                in_range = [(self.xMinRange < xRand.x < self.xMaxRange),
                            (self.yMinRange < xRand.y < self.yMaxRange),
                            (self.zMinRange < xRand.z < self.zMaxRange)]
                if all(in_range):
                    break
        else:
            if biasToNode is not None:
                xRand = self.bias_sampling(biasToNode)
            else:
                xRand = self.uni_sampling()
        return xRand

    def unit_ball_sampling(self):
        u = np.random.normal(0, 1, (1, 3 + 2))
        norm = np.linalg.norm(u, axis = -1, keepdims = True)
        u = u/norm
        return u[0,:3].reshape(3,1) #The first N coordinates are uniform in a unit N ball
    
    def rotation_to_world(self, xStart, xGoal, cMax, cMin):
        r1 = cMax / 2
        r2to3 = np.sqrt(cMax**2 - cMin**2) / 2
        L = np.diag([r1, r2to3, r2to3])
        a1 = np.array([[(xGoal.x - xStart.x) / cMin],
                       [(xGoal.y - xStart.y) / cMin],
                       [(xGoal.z - xStart.z) / cMin],])
        I1 = np.array([[1.0], [0.0], [0.0]])
        M = a1 @ I1.T
        U, _, V_T = np.linalg.svd(M, True, True)
        C = U @ np.diag([1.0, 1.0, np.linalg.det(U) * np.linalg.det(V_T.T)]) @ V_T
        return L, C
    
    def plot_tree(self, path, ax):
        self.plot_single_tree(self.treeVertex, path, ax, obstacle_plot=False)

if __name__ == "__main__":
    np.random.seed(9)
    import matplotlib.pyplot as plt
    from planner_util.extract_path_class import extract_path_class_3d
    from target_localization.pre_record_value import wrap_to_pi , newThetaInit, newThetaApp, newThetaGoal
    from copsim.arm_api import UR5eStateArmCoppeliaSimAPI
    from util.general_util import write_dict_to_file

    # Define pose
    thetaInit = newThetaInit
    thetaGoal = wrap_to_pi(newThetaGoal)
    thetaApp = wrap_to_pi(newThetaApp)
    
    planner = RRTInformedDev(thetaInit, thetaApp, thetaGoal, eta=0.2, maxIteration=5000)
    path = planner.planning()
    print(planner.perfMatrix)
    # write_dict_to_file(planner.perfMatrix, "./planner_dev/result_6d/result_6d_rrtstar.txt")
    print(f"==>> path: \n{path}")

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    planner.plot_tree(path, ax)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()
    
    # time.sleep(3)

    # # play back
    planner.copHandle.start_sim()
    pathX, pathY, pathZ = extract_path_class_3d(path)
    pathX = np.array(pathX)
    pathY = np.array(pathY)
    pathZ = np.array(pathZ)

    armState = UR5eStateArmCoppeliaSimAPI()
    armState.set_goal_joint_value(thetaGoal)
    armState.set_aux_joint_value(thetaApp)
    armState.set_start_joint_value(thetaInit)
    # time.sleep(100)
    for i in range(len(pathX)):
        jointVal = np.array([pathX[i], pathY[i], pathZ[i], planner.joint4Fixed, planner.joint5Fixed, planner.joint6Fixed]).reshape(6, 1)
        planner.copHandle.set_joint_value(jointVal)
        time.sleep(2)
        # triggers next simulation step
        # client.step()

    # stop simulation
    planner.copHandle.stop_sim()