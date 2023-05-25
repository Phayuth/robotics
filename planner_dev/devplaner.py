""" Path Planning Development for 6 dof robot
- Main:    RRT            |   Added
- Variant: Bi Directional |   Added
- Goal Rejection          |   Currently Adding
- Sampling : Bias         |   Added
- Pruning                 |   Currently Adding
- Point Informed Resampling Optimization | Currently Adding (More or Less like STOMP optimazation, currently investigate)
- Vector database collision store | Currently Adding
- Multiple Waypoint ?

"""

import os
import sys

wd = os.path.abspath(os.getcwd())
sys.path.append(str(wd))

import numpy as np
from collision_check_geometry.collision_class import ObjLine2D, intersect_line_v_rectangle


class Node:

    def __init__(self, x, y, z, p, q, r, parent=None, cost=0.0) -> None:
        self.x = x
        self.y = y
        self.z = z
        self.p = p
        self.q = q
        self.r = r
        self.parent = parent
        self.cost = cost


class DevPlanner():

    def __init__(self, robot, taskMapObs, xStart, xApp, xGoal, eta=0.3, maxIteration=1000) -> None:
        # robot and workspace
        self.robot = robot
        self.taskMapObs = taskMapObs

        self.xMinRange = 0
        self.xMaxRange = np.pi
        self.yMinRange = -np.pi
        self.yMaxRange = np.pi
        self.zMinRange = -np.pi
        self.zMaxRange = np.pi
        self.pMinRange = -np.pi
        self.pMaxRange = np.pi
        self.qMinRange = -np.pi
        self.qMaxRange = np.pi
        self.rMinRange = -np.pi
        self.rMaxRange = np.pi

        self.probabilityGoalBias = 0.4
        self.xStart = Node(xStart[0, 0], xStart[1, 0], xStart[2, 0], xStart[3, 0], xStart[4, 0], xStart[5, 0])
        self.xGoal = Node(xGoal[0, 0], xGoal[1, 0], xGoal[2, 0], xGoal[3, 0], xGoal[4, 0], xGoal[5, 0])
        self.xApp = Node(xApp[0, 0], xApp[1, 0], xApp[2, 0], xApp[3, 0], xApp[4, 0], xApp[5, 0])

        # properties of planner
        self.maxIteration = maxIteration
        self.eta = eta
        self.treeVertexStart = [self.xStart]
        self.treeVertexGoal = [self.xGoal]
        self.nearStart = None
        self.nearGoal = None

        # collision database
        self.configSearched = []
        self.collisionState = []

    def planning(self):
        for itera in range(self.maxIteration):
            print(itera)
            xRand = self.bias_sampling()
            xNearestStart = self.nearest_node(self.treeVertexStart, xRand)
            xNearestGoal = self.nearest_node(self.treeVertexGoal, xRand)
            xNewStart = self.steer(xNearestStart, xRand)
            xNewGoal = self.steer(xNearestGoal, xRand)
            xNewStart.parent = xNearestStart
            xNewGoal.parent = xNearestGoal
            if self.is_config_in_collision(xNewStart) or self.is_connect_config_possible(xNearestStart, xNewStart):
                continue
            else:
                self.treeVertexStart.append(xNewStart)

            if self.is_config_in_collision(xNewGoal) or self.is_connect_config_possible(xNearestGoal, xNewGoal):
                continue
            else:
                self.treeVertexGoal.append(xNewGoal)

            if self.if_both_tree_node_near():
                if self.is_connect_config_possible(self.nearStart, self.nearGoal):
                    break

    def search_path(self):
        pathStart = [self.nearStart]
        currentNodeStart = self.nearStart
        while currentNodeStart != self.xStart:
            currentNodeStart = currentNodeStart.parent
            pathStart.append(currentNodeStart)

        pathGoal = [self.nearGoal]
        currentNodeGoal = self.nearGoal
        while currentNodeGoal != self.xGoal:
            currentNodeGoal = currentNodeGoal.parent
            pathGoal.append(currentNodeGoal)

        pathStart.reverse()
        path = pathStart + pathGoal

        return path
    
    def uni_sampling(self):
        x = np.random.uniform(low=self.xMinRange, high=self.xMaxRange)
        y = np.random.uniform(low=self.yMinRange, high=self.yMaxRange)
        z = np.random.uniform(low=self.zMinRange, high=self.zMaxRange)
        p = np.random.uniform(low=self.pMinRange, high=self.pMaxRange)
        q = np.random.uniform(low=self.qMinRange, high=self.qMaxRange)
        r = np.random.uniform(low=self.rMinRange, high=self.rMaxRange)
        xRand = Node(x, y, z, p, q, r)
        return xRand
    
    def bias_sampling(self):
        if np.random.uniform(low=0, high=1.0) < self.probabilityGoalBias:
            xRand = Node(self.xGoal.x, self.xGoal.y, self.xGoal.z, self.xGoal.p, self.xGoal.q, self.xGoal.r)
        else:
            xRand = self.uni_sampling()
        return xRand

    def nearest_node(self, treeVertex, xRand):
        vertexList = []

        for eachVertex in treeVertex:
            distX = xRand.x - eachVertex.x
            distY = xRand.y - eachVertex.y
            distZ = xRand.z - eachVertex.z
            distP = xRand.p - eachVertex.p
            distQ = xRand.q - eachVertex.q
            distR = xRand.r - eachVertex.r
            dist = np.linalg.norm([distX, distY, distZ, distP, distQ, distR])
            vertexList.append(dist)

        minIndex = np.argmin(vertexList)
        xNear = treeVertex[minIndex]

        return xNear

    def steer(self, xNearest, xRand):
        distX = xRand.x - xNearest.x
        distY = xRand.y - xNearest.y
        distZ = xRand.z - xNearest.z
        distP = xRand.p - xNearest.p
        distQ = xRand.q - xNearest.q
        distR = xRand.r - xNearest.r
        dist = np.linalg.norm([distX, distY, distZ, distP, distQ, distR])

        if dist <= self.eta:
            xNew = xRand
        else:
            newX = self.eta * distX + xNearest.x
            newY = self.eta * distY + xNearest.y
            newZ = self.eta * distZ + xNearest.z
            newP = self.eta * distP + xNearest.p
            newQ = self.eta * distQ + xNearest.q
            newR = self.eta * distR + xNearest.r
            xNew = Node(newX, newY, newZ, newP, newQ, newR)
        return xNew

    def if_both_tree_node_near(self):
        for eachVertexStart in self.treeVertexStart:
            for eachVertexGoal in self.treeVertexGoal:
                distX = eachVertexStart.x - eachVertexGoal.x
                distY = eachVertexStart.y - eachVertexGoal.y
                distZ = eachVertexStart.z - eachVertexGoal.z
                distP = eachVertexStart.p - eachVertexGoal.p
                distQ = eachVertexStart.q - eachVertexGoal.q
                distR = eachVertexStart.r - eachVertexGoal.r
                dist = np.linalg.norm([distX, distY, distZ, distP, distQ, distR])
                if dist < self.eta:
                    self.nearStart = eachVertexStart
                    self.nearGoal = eachVertexGoal
                    return True
        return False
    
    def is_config_in_collision(self, xNew):
        theta = np.array([xNew.x, xNew.y, xNew.z, xNew.p, xNew.q, xNew.r]).reshape(6, 1)
        linkPose = self.robot.forward_kinematic(theta, return_link_pos=True)
        linearm1 = ObjLine2D(linkPose[0][0], linkPose[0][1], linkPose[1][0], linkPose[1][1])
        linearm2 = ObjLine2D(linkPose[1][0], linkPose[1][1], linkPose[2][0], linkPose[2][1])
        linearm3 = ObjLine2D(linkPose[2][0], linkPose[2][1], linkPose[3][0], linkPose[3][1])
        linearm4 = ObjLine2D(linkPose[3][0], linkPose[3][1], linkPose[4][0], linkPose[4][1])
        linearm5 = ObjLine2D(linkPose[4][0], linkPose[4][1], linkPose[5][0], linkPose[5][1])
        linearm6 = ObjLine2D(linkPose[5][0], linkPose[5][1], linkPose[6][0], linkPose[6][1])

        # add collsion state to database
        self.configSearched.append(xNew)

        for obs in self.taskMapObs:
            if intersect_line_v_rectangle(linearm1, obs):
                self.collisionState.append(True)
                return True
            if intersect_line_v_rectangle(linearm2, obs):
                self.collisionState.append(True)
                return True
            if intersect_line_v_rectangle(linearm3, obs):
                self.collisionState.append(True)
                return True
            if intersect_line_v_rectangle(linearm4, obs):
                self.collisionState.append(True)
                return True
            if intersect_line_v_rectangle(linearm5, obs):
                self.collisionState.append(True)
                return True
            if intersect_line_v_rectangle(linearm6, obs):
                self.collisionState.append(True)
                return True

        self.collisionState.append(False)

        return False

    def is_connect_config_possible(self, xNearest, xNew):
        distX = xNew.x - xNearest.x
        distY = xNew.y - xNearest.y
        distZ = xNew.z - xNearest.z
        distP = xNew.p - xNearest.p
        distQ = xNew.q - xNearest.q
        distR = xNew.r - xNearest.r
        desiredStep = 10
        rateX = distX / desiredStep
        rateY = distY / desiredStep
        rateZ = distZ / desiredStep
        rateP = distP / desiredStep
        rateQ = distQ / desiredStep
        rateR = distR / desiredStep
        for i in range(1, desiredStep - 1):
            newX = xNearest.x + (rateX * i)
            newY = xNearest.y + (rateY * i)
            newZ = xNearest.z + (rateZ * i)
            newP = xNearest.p + (rateP * i)
            newQ = xNearest.q + (rateQ * i)
            newR = xNearest.r + (rateR * i)
            xNew = Node(newX, newY, newZ, newP, newQ, newR)
            if self.is_config_in_collision(xNew):
                return True
        return False


if __name__ == "__main__":
    np.random.seed(9)
    from robot.planar_sixdof import PlanarSixDof
    from planner_util.extract_path_class import extract_path_class_6d
    import matplotlib.pyplot as plt
    from taskmap_obs import two_near_ee
    from planner_util.plot_util import plot_joint_6d
    from scipy.optimize import curve_fit

    robot = PlanarSixDof()

    thetaInit = np.array([0,0,0,0,0,0]).reshape(6,1)
    thetaGoal = np.array([1,0,0,0,0,0]).reshape(6,1)
    thetaApp = np.array([0,0,0,0,0,0]).reshape(6,1)
    obsList = two_near_ee()

    # plot pre planning
    fig1, ax1 = plt.subplots()
    ax1.set_aspect("equal")
    ax1.set_title("Pre Planning Plot")
    robot.plot_arm(thetaInit, plt_axis=ax1)
    for obs in obsList:
        obs.plot()
    plt.show()

    planner = DevPlanner(robot, obsList, thetaInit, thetaApp, thetaGoal, eta=0.5, maxIteration=1000)
    planner.planning()
    print(planner.configSearched)
    print(planner.collisionState)
    path = planner.search_path()
    pathX, pathY, pathZ, pathP, pathQ, pathR = extract_path_class_6d(path)

    # plot after planning
    fig2, ax2 = plt.subplots()
    ax2.set_aspect("equal")
    ax2.set_title("After Planning Plot")
    for obs in obsList:
        obs.plot()
    robot.plot_arm(thetaInit, plt_axis=ax2)
    for i in range(len(path)):
        robot.plot_arm(np.array([pathX[i], pathY[i], pathZ[i], pathP[i], pathQ[i], pathR[i]]).reshape(6, 1), plt_axis=ax2)
        plt.pause(1)
    plt.show()


    time = np.linspace(0, 1, len(pathX))

    def quintic5deg(x, a, b, c, d, e, f):
        return a*x**5 + b*x**4 + c*x**3 + d*x**2 + e*x * f

    # Fit the line equation
    poptX, pcovX = curve_fit(quintic5deg, time, pathX)
    poptY, pcovY = curve_fit(quintic5deg, time, pathY)
    poptZ, pcovZ = curve_fit(quintic5deg, time, pathZ)
    poptP, pcovP = curve_fit(quintic5deg, time, pathP)
    poptQ, pcovQ = curve_fit(quintic5deg, time, pathQ)
    poptR, pcovR = curve_fit(quintic5deg, time, pathR)

    timeSmooth = np.linspace(0,1,100)
    # plot after planning
    fig3, ax3 = plt.subplots()
    ax3.set_aspect("equal")
    ax3.set_title("After Planning Plot")
    for obs in obsList:
        obs.plot()
    for i in range(timeSmooth.shape[0]):
        robot.plot_arm(np.array([quintic5deg(timeSmooth[i], *poptX),
                                 quintic5deg(timeSmooth[i], *poptY),
                                 quintic5deg(timeSmooth[i], *poptZ),
                                 quintic5deg(timeSmooth[i], *poptP),
                                 quintic5deg(timeSmooth[i], *poptQ),
                                 quintic5deg(timeSmooth[i], *poptR)]).reshape(6, 1), plt_axis=ax3)
        plt.pause(1)
    plt.show()