""" Path Planning for Planar RR with RRT at runtime
"""

import os
import sys

wd = os.path.abspath(os.getcwd())
sys.path.append(str(wd))

import numpy as np
from collision_check_geometry.collision_class import ObjLine2D, intersect_line_v_rectangle


class Node:

    def __init__(self, x, y, parent=None) -> None:
        self.x = x
        self.y = y
        self.parent = parent


class RuntimeRRTBase():

    def __init__(self, robot, taskMapObs, xStart, xApp, xGoal, eta=0.3, maxIteration=1000) -> None:
        # robot and workspace
        self.robot = robot
        self.taskMapObs = taskMapObs
        self.xMinRange = -np.pi
        self.xMaxRange = np.pi
        self.yMinRange = -np.pi
        self.yMaxRange = np.pi
        self.probabilityGoalBias = 0.2
        self.xStart = Node(xStart[0, 0], xStart[1, 0])
        self.xGoal = Node(xGoal[0, 0], xGoal[1, 0])
        self.xApp = Node(xApp[0, 0], xApp[1, 0])

        self.treeVertexStart = [self.xStart]
        self.treeVertexGoal = [self.xApp]
        
        self.treeSwapFlag = True
        self.connectNodeStart = None
        self.connectNodeGoal = None
        self.distGoalToApp = self.distance_between_config(self.xGoal, self.xApp)
        self.maxIteration = maxIteration
        self.eta = eta

    def planning(self):
        itera = self.planner_rrt_connect_app()
        path = self.search_bidirectional_path()
        return path

    def planner_rrt_connect_app(self):  # Method of Expanding toward Each Other (RRT Connect) + approach pose
        for itera in range(self.maxIteration):
            print(itera)
            if self.treeSwapFlag is True: # Init tree side
                xRand = self.uni_sampling()
                xNearest = self.nearest_node(self.treeVertexStart, xRand)
                xNew = self.steer(xNearest, xRand)
                xNew.parent = xNearest

                if not self.is_config_in_collision(xNew) and \
                    not self.is_connect_config_in_collision(xNew.parent, xNew) and \
                    not self.is_config_in_region_of_config(xNew, self.xGoal, self.distGoalToApp) and \
                    not self.is_connect_config_in_region_of_config(xNew.parent, xNew, self.xGoal, self.distGoalToApp):

                    self.treeVertexStart.append(xNew)
                    xNearestPrime = self.nearest_node(self.treeVertexGoal, xNew)
                    xNewPrime = self.steer(xNearestPrime, xNew)
                    xNewPrime.parent = xNearestPrime

                    if not self.is_config_in_collision(xNewPrime) and \
                        not self.is_connect_config_in_collision(xNewPrime.parent, xNewPrime)and \
                        not self.is_config_in_region_of_config(xNewPrime, self.xGoal, self.distGoalToApp) and \
                        not self.is_connect_config_in_region_of_config(xNewPrime.parent, xNewPrime, self.xGoal, self.distGoalToApp):

                        self.treeVertexGoal.append(xNewPrime)

                        while True:
                            xNewPPrime = self.steer(xNewPrime, xNew)
                            xNewPPrime.parent = xNewPrime

                            # if the 2 node meet, then break
                            if self.distance_between_config(xNewPPrime, xNew) < 1e-3:
                                self.connectNodeGoal = xNewPPrime
                                self.connectNodeStart = xNew
                                break

                            # if there is collision then break and if the node and connection of node to parent is inside region
                            if self.is_config_in_collision(xNewPPrime) or \
                                self.is_connect_config_in_collision(xNewPPrime.parent, xNewPPrime)or \
                                self.is_config_in_region_of_config(xNewPPrime, self.xGoal, self.distGoalToApp) or \
                                self.is_connect_config_in_region_of_config(xNewPPrime.parent, xNewPPrime, self.xGoal, self.distGoalToApp):
                                break

                            # if not collision then free to add
                            else:
                                self.treeVertexGoal.append(xNewPPrime)
                                # oh--! we have to update the xNewPrime to xNewPPrime
                                xNewPrime = xNewPPrime

                if self.connectNodeGoal is not None and self.connectNodeStart is not None:
                    break

                self.tree_swap_flag()

            elif self.treeSwapFlag is False: # App tree side
                xRand = self.uni_sampling()
                xNearest = self.nearest_node(self.treeVertexGoal, xRand)
                xNew = self.steer(xNearest, xRand)
                xNew.parent = xNearest

                if not self.is_config_in_collision(xNew) and \
                    not self.is_connect_config_in_collision(xNew.parent, xNew)and \
                    not self.is_config_in_region_of_config(xNew, self.xGoal, self.distGoalToApp) and \
                    not self.is_connect_config_in_region_of_config(xNew.parent, xNew, self.xGoal, self.distGoalToApp):

                    self.treeVertexGoal.append(xNew)
                    xNearestPrime = self.nearest_node(self.treeVertexStart, xNew)
                    xNewPrime = self.steer(xNearestPrime, xNew)
                    xNewPrime.parent = xNearestPrime

                    if not self.is_config_in_collision(xNewPrime) and \
                        not self.is_connect_config_in_collision(xNewPrime.parent, xNewPrime)and \
                        not self.is_config_in_region_of_config(xNewPrime, self.xGoal, self.distGoalToApp) and \
                        not self.is_connect_config_in_region_of_config(xNewPrime.parent, xNewPrime, self.xGoal, self.distGoalToApp):

                        self.treeVertexStart.append(xNewPrime)
                        while True:
                            xNewPPrime = self.steer(xNewPrime, xNew)
                            xNewPPrime.parent = xNewPrime

                            # if the 2 node meet, then break
                            if self.distance_between_config(xNewPPrime, xNew) < 1e-3:
                                self.connectNodeGoal = xNew
                                self.connectNodeStart = xNewPPrime
                                break

                            # if there is collision then break and if the node and connection of node to parent is inside region
                            if self.is_config_in_collision(xNewPPrime) or \
                                self.is_connect_config_in_collision(xNewPPrime.parent, xNewPPrime) or \
                                self.is_config_in_region_of_config(xNewPPrime, self.xGoal, self.distGoalToApp) or \
                                self.is_connect_config_in_region_of_config(xNewPPrime.parent, xNewPPrime, self.xGoal, self.distGoalToApp):
                                break

                            # if not collision then free to add
                            else:
                                self.treeVertexStart.append(xNewPPrime)
                                # oh--! we have to update the xNewPrime to xNewPPrime
                                xNewPrime = xNewPPrime

                if self.connectNodeGoal is not None and self.connectNodeStart is not None:
                    break

                self.tree_swap_flag()

        return itera

    def search_bidirectional_path(self): # return path is [xinit, x1, x2, ..., xapp, xgoal]
        # nearStart, nearGoal = self.is_both_tree_node_near(return_near_node=True)
        starterNode = self.connectNodeStart # nearStart
        goalerNode = self.connectNodeGoal   # nearGoal

        pathStart = [starterNode]
        currentNodeStart = starterNode
        while currentNodeStart.parent is not None:
            currentNodeStart = currentNodeStart.parent
            pathStart.append(currentNodeStart)

        pathGoal = [goalerNode]
        currentNodeGoal = goalerNode
        while currentNodeGoal.parent is not None:
            currentNodeGoal = currentNodeGoal.parent
            pathGoal.append(currentNodeGoal)

        pathStart.reverse()
        path = pathStart + pathGoal + [self.xGoal]

        return path

    def uni_sampling(self):
        x = np.random.uniform(low=self.xMinRange, high=self.xMaxRange)
        y = np.random.uniform(low=self.yMinRange, high=self.yMaxRange)
        xRand = Node(x, y)

        return xRand
    
    def bias_sampling(self):
        if np.random.uniform(low=0, high=1.0) < self.probabilityGoalBias:
            xRand = Node(self.xGoal.x, self.xGoal.y)
        else:
            x = np.random.uniform(low=self.xMinRange, high=self.xMaxRange)
            y = np.random.uniform(low=self.yMinRange, high=self.yMaxRange)
            xRand = Node(x, y)
        return xRand

    def nearest_node(self, treeVertex, xRand):
        vertexList = []

        for eachVertex in treeVertex:
            vertexList.append(self.distance_between_config(xRand, eachVertex))

        minIndex = np.argmin(vertexList)
        xNear = treeVertex[minIndex]

        return xNear
    
    def tree_swap_flag(self):
        if self.treeSwapFlag is True:
            self.treeSwapFlag = False
        elif self.treeSwapFlag is False:
            self.treeSwapFlag = True

    def steer(self, xNearest, xRand):
        distX, distY= self.distance_each_component_between_config(xNearest, xRand)
        dist = np.linalg.norm([distX, distY])
        if dist <= self.eta:
            xNew = Node(xRand.x, xRand.y)
        else:
            dX = (distX/dist) * self.eta
            dY = (distY/dist) * self.eta
            newX = xNearest.x + dX
            newY = xNearest.y + dY
            xNew = Node(newX, newY)

        return xNew
    
    def is_config_in_region_of_config(self, xToCheck, xCenter, radius=None):
        if radius is None:
            radius = self.eta
        if self.distance_between_config(xToCheck, xCenter) < radius:
            return True
        return False

    def is_connect_config_in_region_of_config(self, xToCheckStart, xToCheckEnd, xCenter, radius=None, NumSeg=10):
        if radius is None:
            radius = self.eta
        distX, distY = self.distance_each_component_between_config(xToCheckStart, xToCheckEnd)
        rateX = distX / NumSeg
        rateY = distY / NumSeg
        for i in range(1, NumSeg - 1):
            newX = xToCheckStart.x + (rateX*i)
            newY = xToCheckStart.y + (rateY*i)
            xNew = Node(newX, newY)
            if self.is_config_in_region_of_config(xNew, xCenter, radius):
                return True
        return False

    def is_config_in_collision(self, xNew):
        theta = np.array([xNew.x, xNew.y]).reshape(2, 1)
        linkPose = self.robot.forward_kinematic(theta, return_link_pos=True)
        linearm1 = ObjLine2D(linkPose[0][0], linkPose[0][1], linkPose[1][0], linkPose[1][1])
        linearm2 = ObjLine2D(linkPose[1][0], linkPose[1][1], linkPose[2][0], linkPose[2][1])

        link = [linearm1, linearm2]

        # obstacle collision
        for obs in self.taskMapObs:
            for i in range(len(link)):
                if intersect_line_v_rectangle(link[i], obs):
                    return True
        return False
    
    def is_connect_config_in_collision(self, xNearest, xNew, NumSeg=10):
        distX, distY = self.distance_each_component_between_config(xNearest, xNew)
        rateX = distX / NumSeg
        rateY = distY / NumSeg
        for i in range(1, NumSeg - 1):
            newX = xNearest.x + (rateX*i)
            newY = xNearest.y + (rateY*i)
            xNew = Node(newX, newY)
            if self.is_config_in_collision(xNew):
                return True
        return False
    
    def distance_between_config(self, xStart, xEnd):
        return np.linalg.norm([xStart.x - xEnd.x,
                               xStart.y - xEnd.y])
    
    def distance_each_component_between_config(self, xStart, xEnd):
        return xEnd.x - xStart.x, \
               xEnd.y - xStart.y, \

    def plot_tree(self, path):
        tree = self.treeVertexGoal + self.treeVertexStart
        for vertex in tree:
            plt.scatter(vertex.x, vertex.y, color="sandybrown")
            if vertex.parent == None:
                pass
            else:
                plt.plot([vertex.x, vertex.parent.x], [vertex.y, vertex.parent.y], color="green")

        if path:
                plt.plot([node.x for node in path], [node.y for node in path], color='blue')

        plt.scatter(self.xStart.x, self.xStart.y, color="red")
        plt.scatter(self.xApp.x, self.xApp.y, color="blue")
        plt.scatter(self.xGoal.x, self.xGoal.y, color="yellow")

if __name__ == "__main__":
    np.random.seed(9)
    import matplotlib.pyplot as plt
    # plt.style.use("seaborn")
    from robot.planar_rr import PlanarRR
    from map.taskmap_geo_format import task_rectangle_obs_1
    from planner_util.extract_path_class import extract_path_class_2d
    # from planner_util.plot_util import plot_tree
    from planner_util.coord_transform import circle_plt

    robot = PlanarRR()
    taskMapObs = task_rectangle_obs_1()

    xStart = np.array([0, 0]).reshape(2, 1)
    xGoal = np.array([np.pi/2, 0]).reshape(2, 1)
    # xApp = np.array([0.7993857287349286, 1.428946878381334]).reshape(2, 1) too large
    xApp = np.array([np.pi/2-0.1, 0.2]).reshape(2, 1)
    robot.plot_arm(xStart, plt_basis=True)
    robot.plot_arm(xGoal)
    robot.plot_arm(xApp)
    for obs in taskMapObs:
        obs.plot()
    plt.show()

    planner = RuntimeRRTBase(robot, taskMapObs, xStart, xApp, xGoal, eta=0.1, maxIteration=1000)
    
    # angle = np.linspace(-np.pi, np.pi, 720)
    # obs_space = [[planner.is_config_in_collision(Node(angle[j], angle[i])) for i in range(360)] for j in range(360)]
    # obs_space = np.array(obs_space)
    # plt.imshow(obs_space)
    # plt.show()
    
    path = planner.planning()
    circle_plt(planner.xGoal.x, planner.xGoal.y, planner.distGoalToApp)
    plt.xlim((-np.pi, np.pi))
    plt.ylim((-np.pi, np.pi))
    planner.plot_tree(path)
    plt.show()

    pathx, pathy = extract_path_class_2d(path)

    plt.axes().set_aspect('equal')
    plt.axvline(x=0, c="green")
    plt.axhline(y=0, c="green")
    obs_list = task_rectangle_obs_1()
    for obs in obs_list:
        obs.plot()
    for i in range(len(path)):
        robot.plot_arm(np.array([[pathx[i]], [pathy[i]]]))
        plt.pause(1)
    plt.show()

    # plot joint 
    t = np.linspace(0,5,len(pathx))
    fig1 = plt.figure("Joint 1")
    ax1 = fig1.add_subplot(111)
    ax1.plot(t,pathx, "ro")
    fig2 = plt.figure("Joint 2")
    ax2 = fig2.add_subplot(111)
    ax2.plot(t,pathy,"ro")
    plt.show()

    # # fit joint
    # from scipy.optimize import curve_fit

    # def quintic5deg(x, a, b, c, d, e, f):
    #     return a*x**5 + b*x**4 + c*x**3 + d*x**2 + e*x * f

    # # Fit the quintic5deg equation to the data
    # popt_x, pcov_x = curve_fit(quintic5deg, t, pathx)
    # popt_y, pcov_y = curve_fit(quintic5deg, t, pathy)

    # fig1 = plt.figure("Joint 1")
    # ax1 = fig1.add_subplot(111)
    # ax1.plot(t,pathx, "ro")
    # ax1.plot(t, quintic5deg(t, *popt_x))

    # fig2 = plt.figure("Joint 2")
    # ax2 = fig2.add_subplot(111)
    # ax2.plot(t,pathy,"ro")
    # ax2.plot(t, quintic5deg(t, *popt_y))
    # plt.show()

    # # plot follow fit traj
    # tnew = np.linspace(0,5,100)
    # plt.axes().set_aspect('equal')
    # plt.axvline(x=0, c="green")
    # plt.axhline(y=0, c="green")
    # obs_list = task_rectangle_obs_1()
    # for obs in obs_list:
    #     obs.plot()
    # for ti in tnew:
    #     robot.plot_arm(np.array([[quintic5deg(ti, *popt_x)], [quintic5deg(ti, *popt_y)]]))
    #     plt.pause(0.1)
    # plt.show()