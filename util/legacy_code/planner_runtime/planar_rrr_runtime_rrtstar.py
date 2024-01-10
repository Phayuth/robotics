""" Path Planning for Planar RRR with RRT star at runtime
"""

import os
import sys

wd = os.path.abspath(os.getcwd())
sys.path.append(str(wd))

import matplotlib.pyplot as plt
import numpy as np

from spatial_geometry.spatial_shape import ShapeLine2D, intersect_line_v_rectangle


class Node:

    def __init__(self, x, y, z, parent=None, cost=0.0) -> None:
        self.x = x
        self.y = y
        self.z = z
        self.parent = parent
        self.cost = cost


class RuntimeRRTStar():

    def __init__(self, robot, taskMapObs, xStart, xGoal, eta=0.3, maxIteration=1000) -> None:
        # robot and workspace
        self.robot = robot
        self.taskMapObs = taskMapObs
        self.xMinRange = 0
        self.xMaxRange = np.pi
        self.yMinRange = -np.pi
        self.yMaxRange = np.pi
        self.zMinRange = -np.pi
        self.zMaxRange = np.pi
        self.probabilityGoalBias = 0.2
        self.xStart = Node(xStart[0, 0], xStart[1, 0], xStart[2, 0])
        self.xGoal = Node(xGoal[0, 0], xGoal[1, 0], xGoal[2, 0])

        # properties of planner
        self.maxIteration = maxIteration
        self.eta = eta
        self.treeVertex = [self.xStart]

    def planning(self):
        for itera in range(self.maxIteration):
            print(itera)
            xRand = self.bias_sampling()
            xNearest = self.nearest_node(xRand)
            xNew = self.steer(xNearest, xRand)
            xNew.parent = xNearest
            xNew.cost = xNew.parent.cost + self.cost_line(xNew, xNew.parent)
            if self.is_config_in_collision(xNew) or self.is_connect_config_possible(xNew.parent, xNew):
                continue
            else:
                XNear = self.near(xNew, self.eta)
                xMin = xNew.parent
                cMin = xMin.cost + self.cost_line(xMin, xNew)
                for xNear in XNear:
                    if self.is_connect_config_possible(xNear, xNew):
                        continue

                    cNew = xNear.cost + self.cost_line(xNear, xNew)
                    if cNew < cMin:
                        xMin = xNear
                        cMin = cNew

                xNew.parent = xMin
                xNew.cost = cMin
                self.treeVertex.append(xNew)

                for xNear in XNear:
                    if self.is_connect_config_possible(xNear, xNew):
                        continue
                    cNear = xNear.cost
                    cNew = xNew.cost + self.cost_line(xNew, xNear)
                    if cNew < cNear:
                        xNear.parent = xNew
                        xNear.cost = xNew.cost + self.cost_line(xNew, xNear)

    def search_path(self):
        # XNear = self.near(self.xGoal, self.eta)
        # for xNear in XNear:
        for xNear in self.treeVertex:
            if self.is_connect_config_possible(xNear, self.xGoal):
                continue
            self.xGoal.parent = xNear

            path = [self.xGoal]
            currentNode = self.xGoal

            while currentNode != self.xStart:
                currentNode = currentNode.parent
                path.append(currentNode)

            path.reverse()
            bestPath = path
            cost = sum(i.cost for i in path)

            if cost < sum(j.cost for j in bestPath):
                bestPath = path

        return bestPath

    def uni_sampling(self):
        x = np.random.uniform(low=self.xMinRange, high=self.xMaxRange)
        y = np.random.uniform(low=self.yMinRange, high=self.yMaxRange)
        z = np.random.uniform(low=self.zMinRange, high=self.zMaxRange)
        xRand = Node(x, y, z)

        return xRand

    def bias_sampling(self):
        if np.random.uniform(low=0, high=1.0) < self.probabilityGoalBias:
            xRand = Node(self.xGoal.x, self.xGoal.y, self.xGoal.z)
        else:
            x = np.random.uniform(low=self.xMinRange, high=self.xMaxRange)
            y = np.random.uniform(low=self.yMinRange, high=self.yMaxRange)
            z = np.random.uniform(low=self.zMinRange, high=self.zMaxRange)
            xRand = Node(x, y, z)
        return xRand

    def nearest_node(self, xRand):
        vertexList = []

        for eachVertex in self.treeVertex:
            distX = xRand.x - eachVertex.x
            distY = xRand.y - eachVertex.y
            distZ = xRand.z - eachVertex.z
            dist = np.linalg.norm([distX, distY, distZ])
            vertexList.append(dist)

        minIndex = np.argmin(vertexList)
        xNear = self.treeVertex[minIndex]

        return xNear

    def steer(self, xNearest, xRand):
        distX = xRand.x - xNearest.x
        distY = xRand.y - xNearest.y
        distZ = xRand.z - xNearest.z
        dist = np.linalg.norm([distX, distY, distZ])

        if dist <= self.eta:
            xNew = xRand
        else:
            newX = self.eta * distX + xNearest.x
            newY = self.eta * distY + xNearest.y
            newZ = self.eta * distZ + xNearest.z
            xNew = Node(newX, newY, newZ)
        return xNew

    def near(self, xNew, minStep):
        neighbor = []
        for index, vertex in enumerate(self.treeVertex):
            dist = np.linalg.norm([(xNew.x - vertex.x), (xNew.y - vertex.y), (xNew.z - vertex.z)])
            if dist <= minStep:
                neighbor.append(index)
        return [self.treeVertex[i] for i in neighbor]

    def cost_line(self, xstart, xend):
        return np.linalg.norm([(xstart.x - xend.x), (xstart.y - xend.y), (xstart.z - xend.z)])

    def is_config_in_collision(self, xNew):
        theta = np.array([xNew.x, xNew.y, xNew.z]).reshape(3, 1)
        linkPose = self.robot.forward_kinematic(theta, return_link_pos=True)
        linkPose = robot.forward_kinematic(theta, return_link_pos=True)
        linearm1 = ShapeLine2D(linkPose[0][0], linkPose[0][1], linkPose[1][0], linkPose[1][1])
        linearm2 = ShapeLine2D(linkPose[1][0], linkPose[1][1], linkPose[2][0], linkPose[2][1])
        linearm3 = ShapeLine2D(linkPose[2][0], linkPose[2][1], linkPose[3][0], linkPose[3][1])

        for obs in self.taskMapObs:
            if intersect_line_v_rectangle(linearm1, obs):
                return True
            else:
                if intersect_line_v_rectangle(linearm2, obs):
                    return True
                else:
                    if intersect_line_v_rectangle(linearm3, obs):
                        return True
        return False

    def is_connect_config_possible(self, xNearest, xNew):  # check if connection between 2 node is possible
        distX = xNew.x - xNearest.x
        distY = xNew.y - xNearest.y
        distZ = xNew.z - xNearest.z
        desiredStep = 10
        rateX = distX / desiredStep
        rateY = distY / desiredStep
        rateZ = distZ / desiredStep
        for i in range(1, desiredStep - 1):
            newX = xNearest.x + (rateX * i)
            newY = xNearest.y + (rateY * i)
            newZ = xNearest.z + (rateZ * i)
            xNew = Node(newX, newY, newZ)
            if self.is_config_in_collision(xNew):
                return True
        return False


if __name__ == "__main__":
    np.random.seed(9)
    from scipy.optimize import curve_fit
    from spatial_geometry.spatial_shape import ShapeRectangle
    from map.taskmap_geo_format import task_rectangle_obs_6
    from robot.planar_rrr import PlanarRRR
    from planner_util.coord_transform import circle_plt
    from planner.extract_path_class import extract_path_class_3d
    from planner_util.plot_util import plot_tree_3d

    robot = PlanarRRR()

    # EXPERIMENT 1 - BASIC PLANNING
    # taskMapObs = task_rectangle_obs_6()

    # xStart = np.array([0, 0, 0]).reshape(3, 1)
    # xGoal = np.array([np.pi / 2, 0, 0]).reshape(3, 1)

    # robot.plot_arm(xStart, plt_basis=True)
    # robot.plot_arm(xGoal)
    # for obs in taskMapObs:
    #     obs.plot()
    # plt.show()

    # planner = RuntimeRRTStar(robot, taskMapObs, xStart, xGoal, eta=0.3, maxIteration=1000)
    # planner.planning()
    # path = planner.search_path()

    # pathx, pathy, pathz = extract_path_class_3d(path)
    # print("==>> pathx: ", pathx)
    # print("==>> pathy: ", pathy)
    # print("==>> pathz: ", pathz)

    # plt.axes().set_aspect('equal')
    # plt.axvline(x=0, c="green")
    # plt.axhline(y=0, c="green")
    # for obs in taskMapObs:
    #     obs.plot()
    # for i in range(len(path)):
    #     robot.plot_arm(np.array([pathx[i], pathy[i], pathz[i]]).reshape(3, 1))
    #     plt.pause(0.5)
    # plt.show()

    # EXPERIMENT 2 - Approach Point
    xTarg = 1.8
    yTarg = 0.2
    alphaTarg = 2  # given from grapse pose candidate
    hD = 0.25
    wD = 0.25
    rCrop = 0.1
    phiTarg = alphaTarg - np.pi
    target = np.array([xTarg, yTarg, phiTarg]).reshape(3, 1)

    xTopStart = (rCrop + hD) * np.cos(alphaTarg - np.pi / 2) + xTarg
    yTopStart = (rCrop + hD) * np.sin(alphaTarg - np.pi / 2) + yTarg
    xBotStart = (rCrop) * np.cos(alphaTarg + np.pi / 2) + xTarg
    yBotStart = (rCrop) * np.sin(alphaTarg + np.pi / 2) + yTarg
    recTop = ShapeRectangle(xTopStart, yTopStart, hD, wD, angle=alphaTarg)
    recBot = ShapeRectangle(xBotStart, yBotStart, hD, wD, angle=alphaTarg)
    obsList = [recTop, recBot]
    thetaGoal = robot.inverse_kinematic_geometry(target, elbow_option=0)
    initPose = np.array([[2.5], [0], [0]])
    thetaInit = robot.inverse_kinematic_geometry(initPose, elbow_option=0)

    approachPose = np.array([[rCrop * np.cos(target[2, 0] + np.pi) + target[0, 0]], [rCrop * np.sin(target[2, 0] + np.pi) + target[1, 0]], [target[2, 0]]])
    thetaApp = robot.inverse_kinematic_geometry(approachPose, elbow_option=0)

    robot.plot_arm(thetaGoal, plt_basis=True)
    robot.plot_arm(thetaApp)
    recTop.plot()
    recBot.plot()
    circle_plt(xTarg, yTarg, radius=rCrop)
    plt.show()

    planner = RuntimeRRTStar(robot, obsList, thetaInit, thetaGoal, eta=0.3, maxIteration=1000)
    planner.planning()
    path = planner.search_path()

    pathx, pathy, pathz = extract_path_class_3d(path)
    plt.axes().set_aspect('equal')
    plt.axvline(x=0, c="green")
    plt.axhline(y=0, c="green")
    for obs in obsList:
        obs.plot()
    circle_plt(xTarg, yTarg, radius=rCrop)
    robot.plot_arm(thetaApp)
    for i in range(len(path)):
        robot.plot_arm(np.array([pathx[i], pathy[i], pathz[i]]).reshape(3, 1))
        plt.pause(1)
    plt.show()

    plot_tree_3d(planner.treeVertex, path)
    plt.show()

    # plot joint
    t = np.linspace(0,5,len(pathx))
    fig, axs = plt.subplots(3)
    axs[0].plot(t,pathx, "ro")
    axs[1].plot(t,pathy, "ro")
    axs[2].plot(t,pathz, "ro")
    plt.show()

    # fit joint
    from scipy.optimize import curve_fit

    def quintic5deg(x, a, b, c, d, e, f):
        return a*x**5 + b*x**4 + c*x**3 + d*x**2 + e*x * f

    # Fit the quintic5deg equation to the data
    poptX, pcovX = curve_fit(quintic5deg, t, pathx)
    poptY, pcovY = curve_fit(quintic5deg, t, pathy)
    poptZ, pcovZ = curve_fit(quintic5deg, t, pathz)

    fig2, axs2 = plt.subplots(3)
    axs2[0].plot(t,pathx, "ro")
    axs2[0].plot(t, quintic5deg(t, *poptX))
    axs2[1].plot(t,pathy, "ro")
    axs2[1].plot(t, quintic5deg(t, *poptY))
    axs2[2].plot(t,pathz, "ro")
    axs2[2].plot(t, quintic5deg(t, *poptZ))
    plt.show()

    # plot follow fit traj
    tnew = np.linspace(0,5,100)
    plt.axes().set_aspect('equal')
    plt.axvline(x=0, c="green")
    plt.axhline(y=0, c="green")
    for obs in obsList:
        obs.plot()
    for ti in tnew:
        robot.plot_arm(np.array([[quintic5deg(ti, *poptX)], [quintic5deg(ti, *poptY)], [quintic5deg(ti, *poptZ)]]))
        plt.pause(0.1)
    plt.show()