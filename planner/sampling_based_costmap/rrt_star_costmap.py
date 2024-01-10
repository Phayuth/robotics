import os
import sys

wd = os.path.abspath(os.getcwd())
sys.path.append(str(wd))

import numpy as np
import matplotlib.pyplot as plt
import time


class Node:

    def __init__(self, x: float, y: float, cost: float = 0, parent=None):
        self.x = x
        self.y = y
        self.cost = cost
        self.parent = parent


class RrtstarCostmap():

    def __init__(self, mapClass, xStart: Node, xGoal: Node, distanceWeight: float, obstacleWeight: float, eta: float = None, maxIteration: int = 1000):
        # map properties
        self.mapClass = mapClass
        self.costMap = self.mapClass.costmap
        self.xMinRange = self.mapClass.xmin
        self.xMaxRange = self.mapClass.xmax
        self.yMinRange = self.mapClass.ymin
        self.yMaxRange = self.mapClass.ymax
        self.xStart = Node(xStart[0, 0], xStart[1, 0])
        self.xStart.cost = 0.0
        self.xGoal = Node(xGoal[0, 0], xGoal[1, 0])

        # planner properties
        self.maxIteration = maxIteration
        self.m = self.costMap.shape[0] * self.costMap.shape[1]
        self.r = (2 * (1 + 1/2)**(1 / 2)) * (self.m / np.pi)**(1 / 2)
        self.eta = self.r * (np.log(self.maxIteration) / self.maxIteration)**(1 / 2)
        self.w1 = distanceWeight
        self.w2 = obstacleWeight
        self.nodes = [self.xStart]

        self.sampleTaken = 0
        self.totalIter = 0

        # timing
        self.s = time.time()
        self.e = None
        self.samplingElapsed = 0
        self.addparentElapsed = 0
        self.rewireElapsed = 0

    def uniform_sampling(self) -> Node:
        x = np.random.uniform(low=self.xMinRange, high=self.xMaxRange - 1)
        y = np.random.uniform(low=self.yMinRange, high=self.yMaxRange - 1)
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

    def distance_cost(self, start: Node, end: Node) -> float:
        distanceCost = np.linalg.norm([(start.x - end.x), (start.y - end.y)])
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

        for xNear in self.nodes:
            if self.distance_cost(xNear, xNew) <= self.eta:
                if xNear.cost + self.line_cost(xNear, xNew) < cMin:
                    xMin = xNear
                    cMin = xNear.cost + self.line_cost(xNear, xNew)
            xNew.parent = xMin
            xNew.cost = cMin

        return xNew

    def rewire(self, xNew: Node):
        for xNear in self.nodes:
            if xNear is not xNew.parent:
                if self.distance_cost(xNear, xNew) <= self.eta:  #and self.obstacle_cost(xNew, xNear) < 1
                    if xNew.cost + self.line_cost(xNew, xNear) < xNear.cost:
                        xNear.parent = xNew
                        xNear.cost = xNew.cost + self.line_cost(xNew, xNear)

    def get_path(self) -> list:
        tempPath = []
        path = []
        n = 0
        for i in self.nodes:
            if self.distance_cost(i, self.xGoal) < self.eta:  #5
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

            while i is not self.xStart:
                path.append(i)
                i = i.parent
            path.append(self.xStart)

            self.xGoal.parent = path[0]
            path.insert(0, self.xGoal)

            return path

    def plt_env(self):
        plt.imshow(self.costMap)

        for i in self.nodes:
            if i is not self.xStart:
                plt.plot([i.x, i.parent.x], [i.y, i.parent.y], "b")


    def draw_path(self, path: list):
        for i in path:
            if i is not self.xStart:
                plt.plot([i.x, i.parent.x], [i.y, i.parent.y], "r", linewidth=2.5)

    def planning(self):
        while True:
            timeSamplingStart = time.time()
            while True:
                xRand = self.uniform_sampling()
                self.totalIter += 1
                xNearest = self.nearest(xRand)
                xNew = self.steer(xRand, xNearest)
                b = self.new_check(xNew)
                if b == True:
                    break
                if self.totalIter == self.maxIteration:
                    break
            timeSamplingEnd = time.time()
            self.samplingElapsed += timeSamplingEnd - timeSamplingStart
            if self.totalIter == self.maxIteration:
                break
            self.sampleTaken += 1
            print("==>> self.sampleTaken: ", self.sampleTaken)

            timeAddparentStart = time.time()
            xNew = self.add_parent(xNew, xNearest)
            timeAddparentEnd = time.time()
            self.addparentElapsed += (timeAddparentEnd - timeAddparentStart)
            self.nodes.append(xNew)

            timeRewireStart = time.time()
            self.rewire(xNew)
            timeRewireEnd = time.time()
            self.rewireElapsed += (timeRewireEnd - timeRewireStart)

        self.e = time.time()

    def print_time(self):
        print("total time : ", self.e - self.s, "second")
        print("sampling time : ", self.samplingElapsed, "second", (self.samplingElapsed * 100) / (self.e - self.s), "%")
        print("add_parent time : ", self.addparentElapsed, "second", (self.addparentElapsed * 100) / (self.e - self.s), "%")
        print("rewire time : ", self.rewireElapsed, "second", (self.rewireElapsed * 100) / (self.e - self.s), "%")
        print("total_iteration = ", self.totalIter)
        print("cost : ", self.xGoal.cost)


if __name__ == "__main__":
    from spatial_geometry.mapclass import CostMapLoader, CostMapClass
    np.random.seed(1)

    # SECTION - Experiment 1
    maploader = CostMapLoader.loadsave(maptype="task", mapindex=2)
    maploader.grid_map_probability(size=3)
    mapclass = CostMapClass(maploader=maploader)
    plt.imshow(mapclass.costmap)
    plt.show()
    xStart = np.array([19.5, 110]).reshape(2, 1)
    xGoal = np.array([110, 17]).reshape(2, 1)


    # SECTION - planner
    rrt = RrtstarCostmap(mapclass, xStart, xGoal, distanceWeight=0.5, obstacleWeight=0.5, maxIteration=1000)
    rrt.planning()
    path = rrt.get_path()
    print(f"==>> path: {path}")


    # SECTION - result
    rrt.plt_env()
    rrt.draw_path(path)
    plt.show()