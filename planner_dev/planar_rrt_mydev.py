""" 
Devplanner with RRT Informed with reject goal sampling

"""
import os
import sys
wd = os.path.abspath(os.getcwd())
sys.path.append(str(wd))

import time
import numpy as np
from planner_dev.planar_rrt_component import Node, RRTComponent2D


class RRTInformedStarDev2D(RRTComponent2D):
    def __init__(self, xStart, xApp, xGoal, eta=0.3, maxIteration=1000) -> None:
        super().__init__()
        # start, aux, goal node
        self.xStart = Node(xStart[0, 0], xStart[1, 0])
        self.xGoal = Node(xGoal[0, 0], xGoal[1, 0])
        self.xApp = Node(xApp[0, 0], xApp[1, 0])

        self.eta = eta
        self.maxIteration = maxIteration
        self.treeVertex = [self.xStart]
        self.treeVertexStart = [self.xStart]
        self.treeVertexGoal = [self.xApp]
        self.treeSwapFlag = True
        self.connectNodeStart = None
        self.connectNodeGoal = None
        self.rewireRadius = 0.5
        self.foundInitialPath = False
        self.distGoalToApp = self.distance_between_config(self.xGoal, self.xApp)
        self.XSoln = []
        self.mergeTree = False

    def planning(self):
        timePlanningStart = time.perf_counter_ns()
        itera = self.planner_rrt_connect_informed()
        path = self.search_singledirection_path()
        timePlanningEnd = time.perf_counter_ns()

        # record performance
        self.perfMatrix["Planner Name"] = "Proposed"
        self.perfMatrix["Parameters"]["eta"] = self.eta 
        self.perfMatrix["Parameters"]["Max Iteration"] = self.maxIteration
        self.perfMatrix["Parameters"]["Rewire Radius"] = self.rewireRadius
        self.perfMatrix["Number of Node"] = len(self.treeVertexStart)
        self.perfMatrix["Total Planning Time"] = (timePlanningEnd-timePlanningStart) * 1e-9
        self.perfMatrix["Planning Time Only"] = self.perfMatrix["Total Planning Time"] - self.perfMatrix["KCD Time Spend"]* 1e-9
        self.perfMatrix["Average KCD Time"] = self.perfMatrix["KCD Time Spend"] / self.perfMatrix["Number of Collision Check"]
        
        return path

    def planner_rrt_connect_informed(self):
        for itera in range(self.maxIteration):
            print(itera)
            if self.foundInitialPath is False:
                if self.treeSwapFlag is True: # Init tree side
                    xRand = self.uni_sampling()
                    xNearest = self.nearest_node(self.treeVertexStart, xRand)
                    xNew = self.steer(xNearest, xRand, self.eta)
                    xNew.parent = xNearest
                    xNew.cost = xNew.parent.cost + self.cost_line(xNew, xNew.parent)
                    
                    if not self.is_config_in_collision(xNew) and \
                        not self.is_connect_config_in_collision(xNew.parent, xNew) and \
                        not self.is_config_in_region_of_config(xNew, self.xGoal, self.distGoalToApp) and \
                        not self.is_connect_config_in_region_of_config(xNew.parent, xNew, self.xGoal, self.distGoalToApp):

                        self.treeVertexStart.append(xNew)
                        xNearestPrime = self.nearest_node(self.treeVertexGoal, xNew)
                        xNewPrime = self.steer(xNearestPrime, xNew, self.eta)
                        xNewPrime.parent = xNearestPrime
                        xNewPrime.cost = xNewPrime.parent.cost + self.cost_line(xNewPrime, xNewPrime.parent)

                        if not self.is_config_in_collision(xNewPrime) and \
                            not self.is_connect_config_in_collision(xNewPrime.parent, xNewPrime) and \
                            not self.is_config_in_region_of_config(xNewPrime, self.xGoal, self.distGoalToApp) and \
                            not self.is_connect_config_in_region_of_config(xNewPrime.parent, xNewPrime, self.xGoal, self.distGoalToApp):

                            self.treeVertexGoal.append(xNewPrime)

                            while True:
                                xNewPPrime = self.steer(xNewPrime, xNew, self.eta)
                                xNewPPrime.parent = xNewPrime
                                xNewPPrime.cost = xNewPPrime.parent.cost + self.cost_line(xNewPPrime, xNewPPrime.parent)
                                
                                # if the 2 node meet, then break
                                if self.distance_between_config(xNewPPrime, xNew) < 1e-3:
                                    self.connectNodeGoal = xNewPPrime
                                    self.connectNodeStart = xNew
                                    self.foundInitialPath = True
                                    break # break out sampling and break out of planner 

                                # if there is collision then break and if the node and connection of node to parent is inside region
                                if self.is_config_in_collision(xNewPPrime) or \
                                    self.is_connect_config_in_collision(xNewPPrime.parent, xNewPPrime) or \
                                    self.is_config_in_region_of_config(xNewPPrime, self.xGoal, self.distGoalToApp) or \
                                    self.is_connect_config_in_region_of_config(xNewPPrime.parent, xNewPPrime, self.xGoal, self.distGoalToApp):
                                    break

                                # if not collision then free to add
                                else:
                                    self.treeVertexGoal.append(xNewPPrime)
                                    xNewPrime = xNewPPrime

                    if self.connectNodeGoal is not None and self.connectNodeStart is not None:
                        self.reparent_merge_tree()
                        # break

                    self.tree_swap_flag()

                elif self.treeSwapFlag is False: # App tree side
                    xRand = self.uni_sampling()
                    xNearest = self.nearest_node(self.treeVertexGoal, xRand)
                    xNew = self.steer(xNearest, xRand, self.eta)
                    xNew.parent = xNearest
                    xNew.cost = xNew.parent.cost + self.cost_line(xNew, xNew.parent)

                    if not self.is_config_in_collision(xNew) and \
                        not self.is_connect_config_in_collision(xNew.parent, xNew) and \
                        not self.is_config_in_region_of_config(xNew, self.xGoal, self.distGoalToApp) and \
                        not self.is_connect_config_in_region_of_config(xNew.parent, xNew, self.xGoal, self.distGoalToApp):

                        self.treeVertexGoal.append(xNew)
                        xNearestPrime = self.nearest_node(self.treeVertexStart, xNew)
                        xNewPrime = self.steer(xNearestPrime, xNew, self.eta)
                        xNewPrime.parent = xNearestPrime
                        xNewPrime.cost = xNewPrime.parent.cost + self.cost_line(xNewPrime, xNewPrime.parent)

                        if not self.is_config_in_collision(xNewPrime) and \
                            not self.is_connect_config_in_collision(xNewPrime.parent, xNewPrime) and \
                            not self.is_config_in_region_of_config(xNewPrime, self.xGoal, self.distGoalToApp) and \
                            not self.is_connect_config_in_region_of_config(xNewPrime.parent, xNewPrime, self.xGoal, self.distGoalToApp):

                            self.treeVertexStart.append(xNewPrime)
                            while True:
                                xNewPPrime = self.steer(xNewPrime, xNew, self.eta)
                                xNewPPrime.parent = xNewPrime
                                xNewPPrime.cost = xNewPPrime.parent.cost + self.cost_line(xNewPPrime, xNewPPrime.parent)

                                # if the 2 node meet, then break
                                if self.distance_between_config(xNewPPrime, xNew) < 1e-3:
                                    self.connectNodeGoal = xNew
                                    self.connectNodeStart = xNewPPrime
                                    self.foundInitialPath = True
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
                                    xNewPrime = xNewPPrime

                    if self.connectNodeGoal is not None and self.connectNodeStart is not None:
                        self.reparent_merge_tree()
                        # break

                    self.tree_swap_flag()

            else:
                print("Enter Informed Mode")
                if len(self.XSoln) == 0:
                    cBest = self.xApp.cost
                    cBestPrevious = np.inf
                else:
                    xSolnCost = [xSoln.parent.cost + self.cost_line(xSoln.parent, xSoln) + self.cost_line(xSoln, self.xApp) for xSoln in self.XSoln]
                    xSolnCost = xSolnCost + [self.xApp.cost]
                    print(f"==>> xSolnCost: \n{xSolnCost}")
                    cBest = min(xSolnCost)
                    if cBest < cBestPrevious : # this have nothing to do with planning itself, just for record performance data only
                        self.perfMatrix["Cost Graph"].append((itera, cBest))
                        cBestPrevious = cBest

                xRand = self.informed_sampling(self.xStart, self.xApp, cBest)
                xNearest = self.nearest_node(self.treeVertexStart, xRand)
                xNew = self.steer(xNearest, xRand, self.eta)
                xNew.parent = xNearest
                if self.is_config_in_region_of_config(xNew, self.xGoal, self.distGoalToApp) or \
                    self.is_connect_config_in_region_of_config(xNew.parent, xNew, self.xGoal, self.distGoalToApp):
                    continue
                xNew.cost = xNew.parent.cost + self.cost_line(xNew, xNew.parent)
                if self.is_config_in_collision(xNew) or self.is_connect_config_in_collision(xNew.parent, xNew):
                    continue
                else:
                    XNear = self.near(self.treeVertexStart, xNew, self.rewireRadius)
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
                    self.treeVertexStart.append(xNew)

                    for xNear in XNear:
                        if self.is_connect_config_in_collision(xNear, xNew):
                            continue
                        cNear = xNear.cost
                        cNew = xNew.cost + self.cost_line(xNew, xNear)
                        if cNew < cNear:
                            xNear.parent = xNew
                            xNear.cost = xNew.cost + self.cost_line(xNew, xNear)

                    # in approach region
                    if self.is_config_in_region_of_config(xNew, self.xApp, radius=self.eta):
                        self.XSoln.append(xNew)

        return itera

    def tree_swap_flag(self):
        if self.treeSwapFlag is True:
            self.treeSwapFlag = False
        elif self.treeSwapFlag is False:
            self.treeSwapFlag = True

    def reparent_merge_tree(self): #have to consider parent and cost re update and have to update all node in tree goal as well
        xTobeParent = self.connectNodeStart
        xNow = self.connectNodeGoal
        while True:
            if xNow.parent is None: # xnow is xApp cause it has no parent, so we arrive at xApp, so we update its parent, cost then kill the process
                xNow.parent = xTobeParent
                xNow.cost = xNow.parent.cost + self.cost_line(xNow.parent, xNow)
                self.treeVertexStart.append(xNow)
                break
            xParentSave = xNow.parent
            xNow.parent = xTobeParent
            xNow.cost = xNow.parent.cost + self.cost_line(xNow.parent, xNow)
            self.treeVertexStart.append(xNow)
            xTobeParent = xNow
            xNow = xParentSave

        # for x in self.treeVertexGoal: # update cost of node in treeGoal but we have to update from the main branch first
        #     x.cost = x.parent.cost + self.cost_line(x.parent, x) # not correct yet

    # def search_singledirection_path(self):
    #     for xNear in self.treeVertexStart:
    #         if self.is_connect_config_in_collision(xNear, self.xApp):
    #             continue
    #         self.xApp.parent = xNear

    #         path = [self.xApp]
    #         currentNode = self.xApp

    #         while currentNode != self.xStart:
    #             currentNode = currentNode.parent
    #             path.append(currentNode)

    #         path.reverse()
    #         bestPath = path
    #         cost = sum(i.cost for i in path)

    #         if cost < sum(j.cost for j in bestPath):
    #             bestPath = path

    #     return bestPath + [self.xGoal]

    def search_singledirection_path(self):
        vertexList = []

        for xNear in self.XSoln:
            if self.is_connect_config_in_collision(xNear, self.xApp):
                continue
            self.xApp.parent = xNear

            path = [self.xApp]
            currentNodeStart = self.xApp
            while currentNodeStart.parent is not None:
                currentNodeStart = currentNodeStart.parent
                path.append(currentNodeStart)

            cost = sum(i.cost for i in path)
            vertexList.append(cost)
        
        minIndex = np.argmin(vertexList)
        xBest = self.XSoln[minIndex]

        self.xApp.parent = xBest
        bestPath = [self.xApp]
        currentNodeStart = self.xApp
        while currentNodeStart.parent is not None:
            currentNodeStart = currentNodeStart.parent
            bestPath.append(currentNodeStart)

        bestPath.reverse()

        return bestPath + [self.xGoal]
    
    def search_singledirection_nocost_path(self):
        starterNode = self.xApp

        pathStart = [starterNode]
        currentNodeStart = starterNode
        while currentNodeStart.parent is not None:
            currentNodeStart = currentNodeStart.parent
            pathStart.append(currentNodeStart)

        pathStart.reverse()

        return pathStart

    def informed_sampling(self, xStart, xGoal, cMax):
        cMin = self.distance_between_config(xStart, xGoal)
        print(cMax, cMin)
        xCenter = np.array([[(xStart.x + xGoal.x) / 2],
                            [(xStart.y + xGoal.y) / 2]])

        L, C = self.rotation_to_world(xStart, xGoal, cMax, cMin)

        while True:
            xBall = self.unit_ball_sampling()
            xRand = (C@L@xBall) + xCenter
            xRand = Node(xRand[0, 0], xRand[1, 0])
            in_range = [(self.xMinRange < xRand.x < self.xMaxRange),
                        (self.yMinRange < xRand.y < self.yMaxRange)]
            if all(in_range):
                break
        return xRand

    def unit_ball_sampling(self):
        u = np.random.normal(0, 1, (1, 2 + 2))
        norm = np.linalg.norm(u, axis = -1, keepdims = True)
        u = u/norm
        return u[0,:2].reshape(2,1) #The first N coordinates are uniform in a unit N ball

    def rotation_to_world(self, xStart, xGoal, cMax, cMin):
        r1 = cMax / 2
        r2 = np.sqrt(cMax**2 - cMin**2) / 2
        L = np.diag([r1, r2])

        a1 = np.array([[(xGoal.x - xStart.x) / cMin],
                       [(xGoal.y - xStart.y) / cMin]])
        I1 = np.array([[1.0], [0.0]])
        M = a1 @ I1.T
        U, _, V_T = np.linalg.svd(M, True, True)
        C = U @ np.diag([1.0, np.linalg.det(U) * np.linalg.det(V_T.T)]) @ V_T

        return L, C

    def plot_tree(self, path=None):
        for vertex in self.treeVertexStart:
            if vertex.parent == None:
                pass
            else:
                plt.plot([vertex.x, vertex.parent.x], [vertex.y, vertex.parent.y], color="darkgray")

        for vertex in self.treeVertexGoal:
            if vertex.parent == None:
                pass
            else:
                plt.plot([vertex.x, vertex.parent.x], [vertex.y, vertex.parent.y], color="darkgray")
        
        if True: # for obstacle space 
            jointRange = np.linspace(-np.pi, np.pi, 360)
            xy_points = []
            for theta1 in jointRange:
                for theta2 in jointRange:
                    result = self.robotEnv.check_collision(Node(theta1, theta2))
                    if result is True:
                        xy_points.append([theta1, theta2])

            xy_points = np.array(xy_points)
            plt.plot(xy_points[:, 0], xy_points[:, 1], color='darkcyan', linewidth=0, marker = 'o', markerfacecolor = 'darkcyan', markersize=1.5)

        if path:
            plt.plot([node.x for node in path], [node.y for node in path], color='blue', linewidth=2, marker = 'o', markerfacecolor = 'plum', markersize=5)

        plt.plot(self.xStart.x, self.xStart.y, color="blue", linewidth=0, marker = 'o', markerfacecolor = 'yellow')
        plt.plot(self.xApp.x, self.xApp.y, color="blue", linewidth=0, marker = 'o', markerfacecolor = 'green')
        plt.plot(self.xGoal.x, self.xGoal.y, color="blue", linewidth=0, marker = 'o', markerfacecolor = 'red')


if __name__ == "__main__":
    np.random.seed(9)
    import matplotlib.pyplot as plt
    # import seaborn as sns
    # sns.set_theme()
    # sns.set_context("paper")
    from planner_util.extract_path_class import extract_path_class_2d
    from planner_util.coord_transform import circle_plt
    from util.general_util import write_dict_to_file

    xStart = np.array([0, 0]).reshape(2, 1)
    xGoal = np.array([np.pi/2, 0]).reshape(2, 1)
    xApp = np.array([np.pi/2-0.1, 0.2]).reshape(2, 1)

    # xApp = np.array([1.96055706, -0.77952147]).reshape(2, 1) # aux1 opt1
    # xApp = np.array([1.18103559, 0.77952147]).reshape(2, 1) # aux1 opt2

    # xApp = np.array([1.89042429, -0.71940546]).reshape(2, 1) # aux2 opt1
    # xApp = np.array([1.17101883+0.02, 0.71940546-0.02]).reshape(2, 1) # aux2 opt2

    # xApp3 = np.array([1.97057382, -0.71940546]).reshape(2, 1) # aux3 opt1
    # xApp3 = np.array([1.25116836, 0.71940546]).reshape(2, 1) # aux3 opt2

    # xApp = np.array([np.pi/2-0.1, -0.2]).reshape(2, 1) # not good 
    # xApp = np.array([0.7344, -0.154]).reshape(2, 1) # collision 

    planner = RRTInformedStarDev2D(xStart, xApp, xGoal, eta=0.3, maxIteration=2000)
    planner.robotEnv.robot.plot_arm(xStart, plt_basis=True)
    planner.robotEnv.robot.plot_arm(xGoal)
    planner.robotEnv.robot.plot_arm(xApp)
    # planner.robotEnv.robot.plot_arm(xApp2)
    # planner.robotEnv.robot.plot_arm(xApp3)
    for obs in planner.robotEnv.taskMapObs:
        obs.plot()
    plt.show()

    path = planner.planning()
    print(planner.perfMatrix)
    write_dict_to_file(planner.perfMatrix, "./planner_dev/result_2d/result_2d_proposed_2000.txt")
    fig, ax = plt.subplots()
    fig.set_size_inches(w=3.40067, h=3.40067)
    fig.tight_layout()
    circle_plt(planner.xGoal.x, planner.xGoal.y, planner.distGoalToApp)
    plt.xlim((-np.pi, np.pi))
    plt.ylim((-np.pi, np.pi))
    planner.plot_tree(path)
    plt.show()

    pathx, pathy = extract_path_class_2d(path)

    plt.axes().set_aspect('equal')
    plt.axvline(x=0, c="green")
    plt.axhline(y=0, c="green")
    obs_list = planner.robotEnv.taskMapObs
    for obs in obs_list:
        obs.plot()
    for i in range(len(path)):
        planner.robotEnv.robot.plot_arm(np.array([[pathx[i]], [pathy[i]]]))
        plt.pause(0.3)
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