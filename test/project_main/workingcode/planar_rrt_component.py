import os
import sys
wd = os.path.abspath(os.getcwd())
sys.path.append(str(wd))

import time
import numpy as np
from arm_env2d import RobotArm2DEnvironment

class Node:

    def __init__(self, x, y, parent=None, cost=0.0) -> None:
        self.x = x
        self.y = y
        self.parent = parent
        self.child = []
        self.cost = cost

    def __repr__(self) -> str:
        return f'\nconfig = [{self.x:.7f}, {self.y:.7f}, hasParent = {True if self.parent != None else False}, NumChild = {len(self.child)}]'

class RRTComponent2D():

    def __init__(self) -> None:
        # robot and workspace
        self.robotEnv = RobotArm2DEnvironment()
        
        # joint limit
        self.xMinRange = -np.pi
        self.xMaxRange = np.pi
        self.yMinRange = -np.pi
        self.yMaxRange = np.pi

        # planner properties
        self.probabilityGoalBias = 0.2

        # collision database 
        self.configSearched = []
        self.collisionState = []

        # performance matrix
        self.perfMatrix = {
            "Planner Name": "",
            "Parameters": {"eta": 0.0, "subEta":0.0, "Max Iteration": 0, "Rewire Radius": 0.0},
            "Number of Node": 0,
            "Total Planning Time": 0.0,  # include KCD
            "KCD Time Spend": 0.0,
            "Planning Time Only": 0.0,
            "Number of Collision Check": 0,
            "Average KCD Time": 0.0,
            "Cost Graph": []
        }

    def uni_sampling(self):
        x = np.random.uniform(low=self.xMinRange, high=self.xMaxRange)
        y = np.random.uniform(low=self.yMinRange, high=self.yMaxRange)
        xRand = Node(x, y)
        return xRand
    
    def bias_sampling(self, biasTowardNode):
        if np.random.uniform(low=0, high=1.0) < self.probabilityGoalBias:
            xRand = Node(biasTowardNode.x, biasTowardNode.y)
        else:
            xRand = self.uni_sampling()
        return xRand

    def nearest_node(self, treeVertex, xRand):
        vertexList = []

        for eachVertex in treeVertex:
            vertexList.append(self.distance_between_config(xRand, eachVertex))

        minIndex = np.argmin(vertexList)
        xNear = treeVertex[minIndex]

        return xNear
    
    def steer(self, xNearest, xRand, toDistance):
        distX, distY = self.distance_each_component_between_config(xNearest, xRand)
        dist = np.linalg.norm([distX, distY])
        if dist <= toDistance:
            xNew = Node(xRand.x, xRand.y)
        else:
            dX = (distX/dist) * toDistance
            dY = (distY/dist) * toDistance
            newX = xNearest.x + dX
            newY = xNearest.y + dY
            xNew = Node(newX, newY)

        return xNew
    
    def near_in_radius(self, treeToSearch, xNew, radiusToSearch): 
        # original Near Algorithm from theory but not practical in realworld
        neighbor = []
        for index, vertex in enumerate(treeToSearch):
            dist = np.linalg.norm([(xNew.x - vertex.x),
                                    (xNew.y - vertex.y)])
            if dist <= radiusToSearch:
                neighbor.append(index)
        return [treeToSearch[i] for i in neighbor]

    def near(self, treeToSearch, xNew, radiusToSearch):
        # fixed subset of radius for rewire, to avoid too much rewiring as the original algorithm
        # option : pick the nearest #numTopNearest node nearest to xNew 
        numTopNearest = 10
        neighborIndexandDist = []
        for index, vertex in enumerate(treeToSearch):
            dist = np.linalg.norm([(xNew.x - vertex.x),
                                    (xNew.y - vertex.y)])
            if dist <= radiusToSearch:
                neighborIndexandDist.append([index, dist])
        
        if len(neighborIndexandDist) < numTopNearest:
            return [treeToSearch[i] for i in [item[0] for item in neighborIndexandDist]]
        else:
            sortedDist = sorted(neighborIndexandDist, key=lambda x: x[1])
            indexSorted = [sortedDist[j][0] for j in [i for i in range(numTopNearest)]]
            return [treeToSearch[i] for i in indexSorted]
    
    def cost_line(self, xStart, xEnd):
        return self.distance_between_config(xStart, xEnd)

    def is_collision_and_in_goal_region(self, xNearest, xNew, xCenter, radius): # return True if any of it is true
        if self.is_config_in_region_of_config(xNew, xCenter, radius):
            return True
        elif self.is_connect_config_in_region_of_config(xNearest, xNew, xCenter, radius):
            return True
        elif self.is_config_in_collision(xNew):
            return True
        elif self.is_connect_config_in_collision(xNearest, xNew):
            return True
        else:
            return False

    def is_config_in_region_of_config(self, xToCheck, xCenter, radius):
        if self.distance_between_config(xToCheck, xCenter) < radius:
            return True
        return False
    
    def is_connect_config_in_region_of_config(self, xToCheckStart, xToCheckEnd, xCenter, radius, NumSeg=10):
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
        timeStartKCD = time.perf_counter_ns()
        collisionState = self.robotEnv.check_collision(xNew)
        timeEndKCD = time.perf_counter_ns()
        self.perfMatrix["KCD Time Spend"] += timeEndKCD - timeStartKCD
        self.perfMatrix["Number of Collision Check"] += 1
        return collisionState
    
    def is_connect_config_in_collision_fixed_nseg(self, xNearest, xNew, NumSeg=10):
        # fixed number of Segmented line
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

    def is_connect_config_in_collision(self, xNearest, xNew):
        # dynamic number of Segmented line, but still limit to 10 to avoid long check
        limitedNumSeg = 10
        distX, distY = self.distance_each_component_between_config(xNearest, xNew)
        dist = np.linalg.norm([distX, distY])
        NumSeg = int(np.ceil(dist/self.subEta))
        if NumSeg > limitedNumSeg:
            NumSeg = limitedNumSeg
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
               xEnd.y - xStart.y

    def tree_swap_flag(self):
        if self.treeSwapFlag is True:
            self.treeSwapFlag = False
        elif self.treeSwapFlag is False:
            self.treeSwapFlag = True

    def search_backtrack_single_directional_path(self, backFromNode, attachNode=None): # return path is [xinit, x1, x2, ..., xapp, xgoal]
        pathStart = [backFromNode]
        currentNodeStart = backFromNode
        while currentNodeStart.parent is not None:
            currentNodeStart = currentNodeStart.parent
            pathStart.append(currentNodeStart)

        pathStart.reverse()

        if attachNode:
            return pathStart + [attachNode] # attachNode is Normally xGoal
        else:
            return pathStart

    def search_backtrack_bidirectional_path(self, backFromNodeTa, backFromNodeTb, attachNode=None): # return path is [xinit, x1, x2, ..., xapp, xgoal]
        # backFromNodeTa = self.connectNodeStart # nearStart
        # backFromNodeTb = self.connectNodeGoal  # nearGoal

        pathStart = [backFromNodeTa]
        currentNodeStart = backFromNodeTa
        while currentNodeStart.parent is not None:
            currentNodeStart = currentNodeStart.parent
            pathStart.append(currentNodeStart)

        pathGoal = [backFromNodeTb]
        currentNodeGoal = backFromNodeTb
        while currentNodeGoal.parent is not None:
            currentNodeGoal = currentNodeGoal.parent
            pathGoal.append(currentNodeGoal)

        pathStart.reverse()

        if attachNode:
            return pathStart + pathGoal + [attachNode] # attachNode is Normally xGoal
        else:
            return pathStart + pathGoal

    def search_best_cost_singledirection_path(self, backFromNode, treeVertexList, attachNode=None):
        vertexListCost = [vertex.cost for vertex in treeVertexList]
        costSortedIndex = sorted(range(len(vertexListCost)), key=lambda i: vertexListCost[i])

        for xNearIndex in costSortedIndex:
            if not self.is_connect_config_in_collision(treeVertexList[xNearIndex], backFromNode):
                path = [backFromNode]
                currentNodeStart = treeVertexList[xNearIndex]
                while currentNodeStart.parent is not None:
                    currentNodeStart = currentNodeStart.parent
                    path.append(currentNodeStart)

                path.reverse()

                if attachNode:
                    return path + [attachNode] # attachNode is Normally xGoal
                else:
                    return path

    def plot_single_tree(self, tree, path, axis, obstacle_plot=False):
        for vertex in tree:
            if vertex.parent == None:
                pass
            else:
                axis.plot([vertex.x, vertex.parent.x], [vertex.y, vertex.parent.y], color="darkgray")

        if obstacle_plot is True:  # for obstacle space
            jointRange = np.linspace(-np.pi, np.pi, 360)
            xy_points = []
            for theta1 in jointRange:
                for theta2 in jointRange:
                    result = self.robotEnv.check_collision(Node(theta1, theta2))
                    if result is True:
                        xy_points.append([theta1, theta2])

            xy_points = np.array(xy_points)
            axis.plot(xy_points[:, 0], xy_points[:, 1], color='darkcyan', linewidth=0, marker='o', markerfacecolor='darkcyan', markersize=1.5)

        if path:
            axis.plot([node.x for node in path], [node.y for node in path], color='blue', linewidth=2, marker='o', markerfacecolor='plum', markersize=5)

        axis.plot(self.xStart.x, self.xStart.y, color="blue", linewidth=0, marker='o', markerfacecolor='yellow')
        axis.plot(self.xApp.x, self.xApp.y, color="blue", linewidth=0, marker='o', markerfacecolor='green')
        axis.plot(self.xGoal.x, self.xGoal.y, color="blue", linewidth=0, marker='o', markerfacecolor='red')

    def plot_dual_tree(self, tree1, tree2, path, axis, obstacle_plot=False):
        for vertex in tree1:
            if vertex.parent == None:
                pass
            else:
                axis.plot([vertex.x, vertex.parent.x], [vertex.y, vertex.parent.y], color="darkgray")

        for vertex in tree2:
            if vertex.parent == None:
                pass
            else:
                axis.plot([vertex.x, vertex.parent.x], [vertex.y, vertex.parent.y], color="darkgray")

        if obstacle_plot is True:  # for obstacle space
            jointRange = np.linspace(-np.pi, np.pi, 360)
            xy_points = []
            for theta1 in jointRange:
                for theta2 in jointRange:
                    result = self.robotEnv.check_collision(Node(theta1, theta2))
                    if result is True:
                        xy_points.append([theta1, theta2])

            xy_points = np.array(xy_points)
            axis.plot(xy_points[:, 0], xy_points[:, 1], color='darkcyan', linewidth=0, marker='o', markerfacecolor='darkcyan', markersize=1.5)

        if path:
            axis.plot([node.x for node in path], [node.y for node in path], color='blue', linewidth=2, marker='o', markerfacecolor='plum', markersize=5)

        axis.plot(self.xStart.x, self.xStart.y, color="blue", linewidth=0, marker='o', markerfacecolor='yellow')
        axis.plot(self.xApp.x, self.xApp.y, color="blue", linewidth=0, marker='o', markerfacecolor='green')
        axis.plot(self.xGoal.x, self.xGoal.y, color="blue", linewidth=0, marker='o', markerfacecolor='red')