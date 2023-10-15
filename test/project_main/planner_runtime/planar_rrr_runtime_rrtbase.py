""" Path Planning for Planar RRR with RRT at runtime
"""

import os
import sys

wd = os.path.abspath(os.getcwd())
sys.path.append(str(wd))

import matplotlib.pyplot as plt
import numpy as np

from collision_check_geometry.collision_class import ObjLine2D, intersect_line_v_rectangle


class Node:

    def __init__(self, x, y, z, parent=None) -> None:
        self.x = x
        self.y = y
        self.z = z
        self.parent = parent


class RuntimeRRTBase():

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
        self.xStart = Node(xStart[0, 0], xStart[1, 0], xStart[2, 0])
        self.xGoal = Node(xGoal[0, 0], xGoal[1, 0], xGoal[2, 0])

        # properties of planner
        self.maxIteration = maxIteration
        self.eta = eta
        self.treeVertex = [self.xStart]

    def planning(self):
        for itera in range(self.maxIteration):
            print(itera)
            xRand = self.sampling()
            xNearest = self.nearest_node(xRand)
            xNew = self.steer(xNearest, xRand)
            xNew.parent = xNearest
            if self.is_config_in_collision(xNew) or self.is_connect_config_possible(xNearest, xNew):
                continue
            else:
                self.treeVertex.append(xNew)

    def search_path(self):
        xNearToGoal = self.nearest_node(self.xGoal)
        self.xGoal.parent = xNearToGoal
        path = [self.xGoal]
        currentNode = self.xGoal

        while currentNode != self.xStart:
            currentNode = currentNode.parent
            path.append(currentNode)

        path.reverse()

        return path

    def sampling(self):
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
            dX = (distX/dist) * self.eta
            dY = (distY/dist) * self.eta
            dZ = (distZ/dist) * self.eta
            newX = xNearest.x + dX
            newY = xNearest.y + dY
            newZ = xNearest.z + dZ
            xNew = Node(newX, newY, newZ)
        return xNew

    def is_config_in_collision(self, xNew):
        theta = np.array([xNew.x, xNew.y, xNew.z]).reshape(3, 1)
        linkPose = self.robot.forward_kinematic(theta, return_link_pos=True)
        linkPose = robot.forward_kinematic(theta, return_link_pos=True)
        linearm1 = ObjLine2D(linkPose[0][0], linkPose[0][1], linkPose[1][0], linkPose[1][1])
        linearm2 = ObjLine2D(linkPose[1][0], linkPose[1][1], linkPose[2][0], linkPose[2][1])
        linearm3 = ObjLine2D(linkPose[2][0], linkPose[2][1], linkPose[3][0], linkPose[3][1])

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
    from collision_check_geometry.collision_class import ObjRec
    from map.taskmap_geo_format import task_rectangle_obs_6
    from robot.planar_rrr import PlanarRRR
    from planner_util.coord_transform import circle_plt
    from planner_util.extract_path_class import extract_path_class_3d

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

    # planner = RuntimeRRTBase(robot, taskMapObs, xStart, xGoal, eta=0.3, maxIteration=1000)
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
    #     plt.pause(0.03)
    # plt.show()

    # EXPERIMENT 2 - Virtual obstacle
    # xTarg = 1.8
    # yTarg = 0.2
    # alphaTarg = 2  # given from grapse pose candidate
    # hD = 0.25
    # wD = 0.25
    # rCrop = 0.1
    # phiTarg = alphaTarg - np.pi
    # targetPose = np.array([xTarg, yTarg, phiTarg]).reshape(3, 1)

    # xTopStart = (rCrop + hD) * np.cos(alphaTarg - np.pi / 2) + xTarg
    # yTopStart = (rCrop + hD) * np.sin(alphaTarg - np.pi / 2) + yTarg
    # xBotStart = (rCrop) * np.cos(alphaTarg + np.pi / 2) + xTarg
    # yBotStart = (rCrop) * np.sin(alphaTarg + np.pi / 2) + yTarg
    # recTop = ObjRec(xTopStart, yTopStart, hD, wD, angle=alphaTarg)
    # recBot = ObjRec(xBotStart, yBotStart, hD, wD, angle=alphaTarg)
    # obsList = [recTop, recBot]
    # thetaGoal = robot.inverse_kinematic_geometry(targetPose, elbow_option=0)
    # initPose = np.array([[2.5], [0], [0]])
    # thetaInit = robot.inverse_kinematic_geometry(initPose, elbow_option=0)

    # robot.plot_arm(thetaGoal, plt_basis=True)
    # recTop.plot()
    # recBot.plot()
    # circle_plt(xTarg, yTarg, radius=rCrop)
    # plt.show()

    # planner = RuntimeRRTBase(robot, obsList, thetaInit, thetaGoal, eta=0.3, maxIteration=5000)
    # planner.planning()
    # path = planner.search_path()

    # pathx, pathy, pathz = extract_path_class_3d(path)
    # print("==>> pathx: ", pathx)
    # print("==>> pathy: ", pathy)
    # print("==>> pathz: ", pathz)

    # plt.axes().set_aspect('equal')
    # plt.axvline(x=0, c="green")
    # plt.axhline(y=0, c="green")
    # for obs in obsList:
    #     obs.plot()
    # circle_plt(xTarg, yTarg, radius=rCrop)
    # for i in range(len(path)):
    #     robot.plot_arm(np.array([pathx[i], pathy[i], pathz[i]]).reshape(3, 1))
    #     plt.pause(1)
    # plt.show()

    # EXPERIMENT 3 
    xTarg = 1.8
    yTarg = 0.5
    alphaTarg = 2  # given from grapse pose candidate
    rCrop = 0.1
    phiTarg = alphaTarg - np.pi
    targetPose = np.array([xTarg, yTarg, phiTarg]).reshape(3, 1)
    thetaGoal = robot.inverse_kinematic_geometry(targetPose, elbow_option=0)
    approachPose = np.array([[rCrop * np.cos(targetPose[2, 0] + np.pi) + targetPose[0, 0]], [rCrop * np.sin(targetPose[2, 0] + np.pi) + targetPose[1, 0]], [targetPose[2, 0]]])
    thetaApp = robot.inverse_kinematic_geometry(approachPose, elbow_option=0)
    initPose = np.array([[2.5], [0], [0]])
    thetaInit = robot.inverse_kinematic_geometry(initPose, elbow_option=0)
    obsList = []
    robot.plot_arm(thetaGoal, plt_basis=True)
    robot.plot_arm(thetaApp)
    robot.plot_arm(thetaInit)
    circle_plt(xTarg, yTarg, rCrop)
    plt.show()

    planner = RuntimeRRTBase(robot, obsList, thetaInit, thetaApp, eta=0.3, maxIteration=5000)
    planner.planning()
    path = planner.search_path()

    pathx, pathy, pathz = extract_path_class_3d(path)
    plt.axes().set_aspect('equal')
    plt.axvline(x=0, c="green")
    plt.axhline(y=0, c="green")
    for obs in obsList:
        obs.plot()
    circle_plt(xTarg, yTarg, radius=rCrop)
    for i in range(len(path)):
        robot.plot_arm(np.array([pathx[i], pathy[i], pathz[i]]).reshape(3, 1))
        plt.pause(1)
    plt.show()