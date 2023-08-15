""" 
Devplanner with RRT Star with reject goal sampling

"""
import os
import sys
wd = os.path.abspath(os.getcwd())
sys.path.append(str(wd))

import time
import numpy as np
from planner_dev.planar_rrt_component import Node, RRTComponent2D


class RRTStarDev2D(RRTComponent2D):
    def __init__(self, xStart, xApp, xGoal, eta=0.3, maxIteration=1000) -> None:
        super().__init__()
        # start, aux, goal node
        self.xStart = Node(xStart[0, 0], xStart[1, 0])
        self.xGoal = Node(xGoal[0, 0], xGoal[1, 0])
        self.xApp = Node(xApp[0, 0], xApp[1, 0])

        self.eta = eta
        self.maxIteration = maxIteration
        self.treeVertex = [self.xStart]
        self.distGoalToApp = self.distance_between_config(self.xGoal, self.xApp)
        self.rewireRadius = 0.5
        self.XInGoalRegion = []
        # self.m = (self.xMaxRange - self.xMinRange) * (self.yMaxRange - self.yMinRange)
        # self.radius = (2 * (1 + 1/2)**(1 / 2)) * (self.m / np.pi)**(1 / 2)
        # self.eta = self.radius * (np.log(self.maxIteration) / self.maxIteration)**(1 / 2)

    def planning(self):
        timePlanningStart = time.perf_counter_ns()
        itera = self.planner_rrt_star()
        path = self.search_singledirection_path()
        timePlanningEnd = time.perf_counter_ns()

        # record performance
        self.perfMatrix["Planner Name"] = "RRT Star"
        self.perfMatrix["Parameters"]["eta"] = self.eta 
        self.perfMatrix["Parameters"]["Max Iteration"] = self.maxIteration
        self.perfMatrix["Parameters"]["Rewire Radius"] = self.rewireRadius
        self.perfMatrix["Number of Node"] = len(self.treeVertex)
        self.perfMatrix["Total Planning Time"] = (timePlanningEnd-timePlanningStart) * 1e-9
        self.perfMatrix["Planning Time Only"] = self.perfMatrix["Total Planning Time"] - self.perfMatrix["KCD Time Spend"]* 1e-9
        self.perfMatrix["Average KCD Time"] = self.perfMatrix["KCD Time Spend"] / self.perfMatrix["Number of Collision Check"]
        
        return path

    def planner_rrt_star(self):
        for itera in range(self.maxIteration):
            print(itera)
            # this have nothing to do with planning itself, just for record performance data only
            if len(self.XInGoalRegion) == 0:
                cBest = np.inf
                cBestPrevious = np.inf
            else:
                xSolnCost = [xSoln.parent.cost + self.cost_line(xSoln.parent, xSoln) + self.cost_line(xSoln, self.xApp) for xSoln in self.XInGoalRegion]
                print(f"==>> xSolnCost: \n{xSolnCost}")
                cBest = min(xSolnCost)
                if cBest < cBestPrevious :
                    self.perfMatrix["Cost Graph"].append((itera, cBest))
                    cBestPrevious = cBest

            xRand = self.uni_sampling()
            xNearest = self.nearest_node(self.treeVertex, xRand)
            xNew = self.steer(xNearest, xRand, self.eta)
            xNew.parent = xNearest
            if self.is_config_in_region_of_config(xNew, self.xGoal, self.distGoalToApp) or \
                self.is_connect_config_in_region_of_config(xNearest, xNew, self.xGoal, self.distGoalToApp):
                continue
            xNew.cost = xNew.parent.cost + self.cost_line(xNew, xNew.parent)
            if self.is_config_in_collision(xNew) or self.is_connect_config_in_collision(xNew.parent, xNew):
                continue
            else:
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

                # in goal region, this have nothing to do with planning itself, just for record performance data only
                if self.is_config_in_region_of_config(xNew, self.xApp, radius=self.eta):
                    self.XInGoalRegion.append(xNew)

        return itera

    # def search_singledirection_path(self):
    #     XNear = self.near(self.treeVertex, self.xApp, self.rewireRadius)
    #     for xNear in XNear:
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

        for xNear in self.XInGoalRegion:
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
        xBest = self.XInGoalRegion[minIndex]

        self.xApp.parent = xBest
        bestPath = [self.xApp]
        currentNodeStart = self.xApp
        while currentNodeStart.parent is not None:
            currentNodeStart = currentNodeStart.parent
            bestPath.append(currentNodeStart)

        bestPath.reverse()

        return bestPath + [self.xGoal]
    
    def plot_tree(self, path):
        for vertex in self.treeVertex:
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

    planner = RRTStarDev2D(xStart, xApp, xGoal, eta=0.3, maxIteration=2000)
    planner.robotEnv.robot.plot_arm(xStart, plt_basis=True)
    planner.robotEnv.robot.plot_arm(xGoal)
    planner.robotEnv.robot.plot_arm(xApp)
    for obs in planner.robotEnv.taskMapObs:
        obs.plot()
    plt.show()

    path = planner.planning()
    print(planner.perfMatrix)
    write_dict_to_file(planner.perfMatrix, "./planner_dev/result_2d/result_2d_rrtstar.txt")
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