""" Informed RRT star 2D.
Map Type : Costmap
Sampling Method : Informed Sampling + Uniform/Bias
Collsion : Costmap Collsion check
Path Searcher : Cosider all nodes in the radius of xGoal. -> Calculate Cost to xStart -> Choose best with the lowest cost.
"""

import os
import sys

wd = os.path.abspath(os.getcwd())
sys.path.append(str(wd))

import numpy as np
import matplotlib.pyplot as plt
import time


class Node:

    def __init__(self, x: float, y: float, cost: float = 0, costr: float = 0, parent=None):
        self.x = x
        self.y = y
        self.cost = cost
        self.costr = costr
        self.parent = parent


class InfmrrtstarCostmapUnisampling:

    def __init__(self, mapClass, xInit: Node, xGoal: Node, distanceWeight: float, obstacleWeight: float, eta: float = None, maxIteration: int = 1000):
        # map properties
        self.mapClass = mapClass
        self.costMap = self.mapClass.costmap
        self.xMinRange = self.mapClass.xmin
        self.xMaxRange = self.mapClass.xmax
        self.yMinRange = self.mapClass.ymin
        self.yMaxRange = self.mapClass.ymax
        self.xInit = Node(xInit[0, 0], xInit[1, 0])
        self.xGoal = Node(xGoal[0, 0], xGoal[1, 0])
        self.nodes = [self.xInit]

        # planner properties
        self.maxIteration = maxIteration
        self.m = self.costMap.shape[0] * self.costMap.shape[1]
        self.r = (2 * (1 + 1/2)**(1 / 2)) * (self.m / np.pi)**(1 / 2)
        self.eta = self.r * (np.log(self.maxIteration) / self.maxIteration)**(1 / 2)
        self.w1 = distanceWeight
        self.w2 = obstacleWeight
        self.XSoln = []

        self.sampleTaken = 0
        self.totalIter = 0

        # timing
        self.s = time.time()
        self.e = None
        self.samplingElapsed = 0
        self.addparentElapsed = 0
        self.rewireElapsed = 0

    def uniform_sampling(self) -> Node:
        x = np.random.uniform(low=0, high=self.costMap.shape[0] - 1)  # must have -1 when implement with costmap
        y = np.random.uniform(low=0, high=self.costMap.shape[1] - 1)
        xRand = Node(x, y)
        return xRand

    def bias_sampling(self) -> Node:
        row = self.costMap.shape[1]
        p = np.ravel(self.costMap) / np.sum(self.costMap)
        xSample = np.random.choice(len(p), p=p)
        x = xSample // row
        y = xSample % row
        x = np.random.uniform(low=x - 0.5, high=x + 0.5)
        y = np.random.uniform(low=y - 0.5, high=y + 0.5)
        xRand = Node(x, y)
        return xRand

    def sampling(self, xStart, xGoal, cMax):
        if cMax < np.inf:
            cMin = self.distance_cost(xStart, xGoal)
            print(cMax, cMin)
            xCenter = np.array([(xStart.x + xGoal.x) / 2, (xStart.y + xGoal.y) / 2]).reshape(2, 1)
            C = self.rotation_to_world_frame(xStart, xGoal)
            r1 = cMax / 2
            r2 = np.sqrt(cMax**2 - cMin**2) / 2
            L = np.diag([r1, r2])
            while True:
                xBall = self.unit_ball_sampling()
                xRand = (C@L@xBall) + xCenter
                xRand = Node(xRand[0, 0], xRand[1, 0])
                if (0 < xRand.x < self.costMap.shape[0] - 1) and (0 < xRand.y < self.costMap.shape[1] - 1):  # check if inside configspace
                    break
        else:
            xRand = self.bias_sampling()
        return xRand

    def unit_ball_sampling(self):
        r = np.random.uniform(low=0, high=1)
        theta = np.random.uniform(low=0, high=2 * np.pi)
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        return np.array([[x], [y]]).reshape(2, 1)

    def ingoal_region(self, xNew):
        if np.linalg.norm([self.xGoal.x - xNew.x, self.xGoal.y - xNew.y]) <= 50:  # self.eta:
            return True
        else:
            return False

    def rotation_to_world_frame(self, xStart, xGoal):
        theta = np.arctan2((xGoal.y - xStart.y), (xGoal.x - xStart.x))
        R = np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]]).T
        return R

    def distance_cost(self, start: Node, end: Node) -> float:
        distanceCost = np.linalg.norm([start.x - end.x, start.y - end.y])
        return distanceCost

    def obstacle_cost(self, start: float, end: float) -> float:
        segLength = 1
        segPoint = int(np.ceil(self.distance_cost(start, end) / segLength))

        value = 0
        if segPoint > 1:
            v = np.array([end.x - start.x, end.y - start.y]) / (segPoint)

            for i in range(segPoint + 1):
                seg = np.array([start.x, start.y]) + i*v
                seg = np.around(seg)
                if 1 - self.costMap[int(seg[1]), int(seg[0])] == 1:
                    cost = 1e10
                    return cost
                else:
                    value += 1 - self.costMap[int(seg[1]), int(seg[0])]
            cost = value / (segPoint+1)
            return cost
        else:
            value = (self.costMap[int(start.y), int(start.x)] + self.costMap[int(end.y), int(end.x)])
            cost = value / 2
            return cost

    def line_cost(self, start: Node, end: Node) -> float:
        cost = self.w1 * (self.distance_cost(start, end) / (self.eta)) + self.w2 * (self.obstacle_cost(start, end))
        return cost

    def nearest(self, xRand: Node) -> Node:
        vertex = []
        i = 0
        for xNear in self.nodes:
            dist = self.distance_cost(xNear, xRand)
            vertex.append([dist, i, xNear])
            i += 1
        vertex.sort()
        xNearest = vertex[0][2]
        return xNearest

    def steer(self, xRand: Node, xNearest: Node) -> Node:
        d = self.distance_cost(xRand, xNearest)

        if d < self.eta:
            xNew = Node(xRand.x, xRand.y)
        else:
            newX = xNearest.x + self.eta * ((xRand.x - xNearest.x) / d)
            newY = xNearest.y + self.eta * ((xRand.y - xNearest.y) / d)
            xNew = Node(newX, newY)
        return xNew

    def exist_check(self, xNew: Node) -> bool:
        for xNear in self.nodes:
            if xNew.x == xNear.x and xNew.y == xNear.y:
                return False
            else:
                return True

    def new_check(self, xNew: Node) -> bool:
        xPob = np.array([xNew.x, xNew.y])
        xPob = np.around(xPob)

        if xPob[0] >= self.costMap.shape[0]:
            xPob[0] = self.costMap.shape[0] - 1
        if xPob[1] >= self.costMap.shape[1]:
            xPob[1] = self.costMap.shape[1] - 1

        xPob = self.costMap[int(xPob[1]), int(xPob[0])]
        p = np.random.uniform(0, 1)

        if xPob > p and self.exist_check(xNew):
            return True
        else:
            return False

    def add_parent(self, xNew: Node, xNearest: Node) -> Node:
        xMin = xNearest
        cMin = xMin.cost + self.line_cost(xMin, xNew)
        cMinr = xMin.costr + self.distance_cost(xMin, xNew)  # real cost

        for xNear in self.nodes:
            if self.distance_cost(xNear, xNew) <= self.eta:
                if xNear.cost + self.line_cost(xNear, xNew) < cMin:
                    xMin = xNear
                    cMin = xNear.cost + self.line_cost(xNear, xNew)
                    cMinr = xNear.costr + self.distance_cost(xNear, xNew)
            xNew.parent = xMin
            xNew.cost = cMin
            xNew.costr = cMinr

        return xNew

    def rewire(self, xNew: Node):
        for xNear in self.nodes:
            if xNear is not xNew.parent:
                if (self.distance_cost(xNear, xNew) <= self.eta):  # and self.obstacle_cost(xNew, xNear) < 1
                    if xNew.cost + self.line_cost(xNew, xNear) < xNear.cost:
                        xNear.parent = xNew
                        xNear.cost = xNew.cost + self.line_cost(xNew, xNear)
                        xNear.costr = xNew.costr + self.distance_cost(xNew, xNear)

    def get_path(self) -> list:
        tempPath = []
        path = []
        n = 0
        for i in self.nodes:
            if self.distance_cost(i, self.xGoal) < self.eta:  # 5
                cost = i.cost + self.line_cost(self.xGoal, i)
                tempPath.append([cost, n, i])
                n += 1
        tempPath.sort()

        if tempPath == []:
            print("cannot find path")
            return None

        else:
            closestNode = tempPath[0][2]
            i = closestNode
            self.xGoal.cost = tempPath[0][0]

            while i is not self.xInit:
                path.append(i)
                i = i.parent
            path.append(self.xInit)

            self.xGoal.parent = path[0]
            path.insert(0, self.xGoal)

            return path

    def draw_tree(self):
        for i in self.nodes:
            if i is not self.xInit:
                plt.plot([i.x, i.parent.x], [i.y, i.parent.y], "b")

    def draw_path(self, path: list):
        for i in path:
            if i is not self.xInit:
                plt.plot([i.x, i.parent.x], [i.y, i.parent.y], "r", linewidth=2.5)

    def planning(self):
        while True:
            cBest = np.inf
            while True:
                for xSoln in self.XSoln:
                    cBest = xSoln.parent.costr + self.distance_cost(xSoln.parent, xSoln) + self.distance_cost(xSoln, self.xGoal)
                    if xSoln.parent.costr + self.distance_cost(xSoln.parent, xSoln) + self.distance_cost(xSoln, self.xGoal) < cBest:
                        cBest = xSoln.parent.costr + self.distance_cost(xSoln.parent, xSoln) + self.distance_cost(xSoln, self.xGoal)

                xRand = self.sampling(self.xInit, self.xGoal, cBest)
                self.totalIter += 1
                xNearest = self.nearest(xRand)
                xNew = self.steer(xRand, xNearest)
                b = self.new_check(xNew)
                if b == True:
                    break
                if self.totalIter == self.maxIteration:
                    break
            if self.totalIter == self.maxIteration:
                break
            self.sampleTaken += 1
            print("==>> self.sampleTaken: ", self.sampleTaken)

            xNew = self.add_parent(xNew, xNearest)
            self.nodes.append(xNew)

            self.rewire(xNew)

            # in goal region
            if self.ingoal_region(xNew):
                self.XSoln.append(xNew)

        # record end of planner loop
        self.e = time.time()

    def print_time(self):
        print("Total Time : ", self.e - self.s, "second")
        print("Sampling Time : ", self.samplingElapsed, "second", (self.samplingElapsed * 100) / (self.e - self.s), "%")
        print("Add Parent Time : ", self.addparentElapsed, "second", (self.addparentElapsed * 100) / (self.e - self.s), "%")
        print("Rewire Time : ", self.rewireElapsed, "second", (self.rewireElapsed * 100) / (self.e - self.s), "%")
        print("Total Iteration = ", self.totalIter)
        print("Cost : ", self.xGoal.cost)


if __name__ == "__main__":
    from map.mapclass import CostMapLoader, CostMapClass
    np.random.seed(0)

    # SECTION - Experiment 1
    maploader = CostMapLoader.loadsave(maptype="task", mapindex=1)
    mapclass = CostMapClass(maploader=maploader, maprange=[[-np.pi, np.pi], [-np.pi, np.pi]])
    plt.imshow(mapclass.costmap)
    plt.show()
    xInit = np.array([24, 12]).reshape(2, 1)
    xGoal = np.array([1.20, 13.20]).reshape(2, 1)

    # SECTION - planner
    distanceWeight = 0.5
    obstacleWeight = 0.5
    rrt = InfmrrtstarCostmapUnisampling(mapclass, xInit, xGoal, distanceWeight, obstacleWeight, maxIteration=500)
    rrt.planning()
    path = rrt.get_path()

    # SECTION - result
    plt.imshow(mapclass.costmap)
    rrt.draw_tree()
    rrt.draw_path(path)
    plt.show()
