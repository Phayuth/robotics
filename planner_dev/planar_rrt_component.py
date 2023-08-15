import os
import sys
wd = os.path.abspath(os.getcwd())
sys.path.append(str(wd))

import time
import numpy as np
from arm_env2d import RobotArm2DEnvironment

class Node:

    def __init__(self, x, y, parent=None, child=[], cost=0.0) -> None:
        self.x = x
        self.y = y
        self.parent = parent
        self.child = child
        self.cost = cost

    def __repr__(self) -> str:
        return f'\nconfig = [{self.x:.7f}, {self.y:.7f}, hasParent = {True if self.parent != None else False}]'

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
            "Parameters": {"eta": 0.0, "Max Iteration": 0, "Rewire Radius": 0.0},
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
    
    def near(self, treeToSearch, xNew, radiusToSearch):
        neighbor = []
        for index, vertex in enumerate(treeToSearch):
            dist = np.linalg.norm([(xNew.x - vertex.x),
                                    (xNew.y - vertex.y)])
            if dist <= radiusToSearch:
                neighbor.append(index)
        return [treeToSearch[i] for i in neighbor]
    
    def cost_line(self, xStart, xEnd):
        return self.distance_between_config(xStart, xEnd)
    
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

    # def plot_tree(self, path):
    #     tree = self.treeVertexGoal + self.treeVertexStart
    #     for vertex in tree:
    #         plt.scatter(vertex.x, vertex.y, color="sandybrown")
    #         if vertex.parent == None:
    #             pass
    #         else:
    #             plt.plot([vertex.x, vertex.parent.x], [vertex.y, vertex.parent.y], color="green")

    #     if path:
    #             plt.plot([node.x for node in path], [node.y for node in path], color='blue')

    #     plt.scatter(self.xStart.x, self.xStart.y, color="red")
    #     plt.scatter(self.xApp.x, self.xApp.y, color="blue")
    #     plt.scatter(self.xGoal.x, self.xGoal.y, color="yellow")