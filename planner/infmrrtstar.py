""" Informed RRT star 2D.
Map Type : Continuous configuration space 2D
Sampling Method : Informed Sampling + Uniform
Collsion : Geometry based
Path Searcher : Cosider all nodes in the radius of xGoal. -> Calculate Cost to xStart -> Choose best with the lowest cost.
"""

import os
import sys

wd = os.path.abspath(os.getcwd())
sys.path.append(str(wd))

import numpy as np
import matplotlib.pyplot as plt
from collision_check_geometry.collision_class import ObjLine2D, ObjPoint2D, intersect_point_v_rectangle, intersect_line_v_rectangle


class Node:

    def __init__(self, x, y, parent=None, cost=0.0) -> None:
        self.x = x
        self.y = y
        self.parent = parent
        self.cost = cost


class InfmRrtstar:

    def __init__(self, mapclass, xStart, xGoal, eta=None, maxIteration=1000) -> None:
        # map properties
        self.mapClass = mapclass
        self.xMinRange = self.mapClass.xmin
        self.xMaxRange = self.mapClass.xmax
        self.yMinRange = self.mapClass.ymin
        self.yMaxRange = self.mapClass.ymax
        self.xStart = Node(xStart[0, 0], xStart[1, 0])
        self.xStart.cost = 0.0
        self.xGoal = Node(xGoal[0, 0], xGoal[1, 0])

        if mapclass.__class__.__name__ == "CostMapClass":
            self.obs = self.mapClass.costmap2geo()
        else:
            self.obs = self.mapClass.obj

        # properties of planner
        self.maxIteration = maxIteration
        self.m = (self.xMaxRange - self.xMinRange) * (self.yMaxRange - self.yMinRange)
        self.radius = (2 * (1 + 1/2)**(1 / 2)) * (self.m / np.pi)**(1 / 2)
        self.eta = self.radius * (np.log(self.maxIteration) / self.maxIteration)**(1 / 2)

        # start with a tree vertex have start node and empty branch
        self.treeVertex = [self.xStart]
        self.XSoln = []

    def planning(self):
        cBest = np.inf
        for itera in range(self.maxIteration):
            print(itera)
            for xSoln in self.XSoln:
                cBest = xSoln.parent.cost + self.cost_line(xSoln.parent, xSoln) + self.cost_line(xSoln, self.xGoal)
                if xSoln.parent.cost + self.cost_line(xSoln.parent, xSoln) + self.cost_line(xSoln, self.xGoal) < cBest:
                    cBest = xSoln.parent.cost + self.cost_line(xSoln.parent, xSoln) + self.cost_line(xSoln, self.xGoal)

            xRand = self.sampling(self.xStart, self.xGoal, cBest)

            xNearest = self.nearest_node(xRand)
            xNew = self.steer(xNearest, xRand)
            xNew.parent = xNearest
            xNew.cost = xNew.parent.cost + self.cost_line(xNew, xNew.parent)  # add the vertex new, which we have to calculate the cost and add parent as well
            if self.collision_check_node(xNew) or self.collision_check_line(xNew.parent, xNew):
                continue
            else:
                XNear = self.near(xNew, self.eta)
                xMin = xNew.parent
                cMin = xMin.cost + self.cost_line(xMin, xNew)
                for xNear in XNear:
                    if self.collision_check_line(xNear, xNew):
                        continue

                    cNew = xNear.cost + self.cost_line(xNear, xNew)
                    if cNew < cMin:
                        xMin = xNear
                        cMin = cNew

                xNew.parent = xMin
                xNew.cost = cMin
                self.treeVertex.append(xNew)

                for xNear in XNear:
                    if self.collision_check_line(xNear, xNew):
                        continue
                    cNear = xNear.cost
                    cNew = xNew.cost + self.cost_line(xNew, xNear)
                    if cNew < cNear:
                        xNear.parent = xNew
                        xNear.cost = xNew.cost + self.cost_line(xNew, xNear)

                # in goal region
                if self.ingoal_region(xNew):
                    self.XSoln.append(xNew)

    def search_path(self):
        for xBest in self.XSoln:
            if self.collision_check_line(xBest, self.xGoal):
                continue
            self.xGoal.parent = xBest

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

    def sampling(self, xStart, xGoal, cMax):
        if cMax < np.inf:
            cMin = self.cost_line(xStart, xGoal)
            print(cMax, cMin)
            xCenter = np.array([(xStart.x + xGoal.x) / 2, (xStart.y + xGoal.y) / 2]).reshape(2, 1)
            C = self.rotation_to_world_frame(xStart, xGoal)
            r1 = cMax / 2
            r2 = np.sqrt(cMax**2 - cMin**2) / 2
            L = np.diag([r1, r2])
            while True:
                xBall = self.unitballsampling()
                xRand = (C@L@xBall) + xCenter
                xRand = Node(xRand[0, 0], xRand[1, 0])
                if (self.xMinRange < xRand.x < self.xMaxRange) and (self.yMinRange < xRand.y < self.yMaxRange):  # check if outside configspace
                    break
        else:
            xRand = self.unisampling()
        return xRand

    def unitballsampling(self):
        r = np.random.uniform(low=0, high=1)
        theta = np.random.uniform(low=0, high=2 * np.pi)
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        return np.array([[x], [y]]).reshape(2, 1)

    def unisampling(self):
        x = np.random.uniform(low=self.xMinRange, high=self.xMaxRange)
        y = np.random.uniform(low=self.yMinRange, high=self.yMaxRange)
        xRand = Node(x, y)
        return xRand

    def ingoal_region(self, xNew):
        if np.linalg.norm([self.xGoal.x - xNew.x, self.xGoal.y - xNew.y]) <= 1:  # self.eta:
            return True
        else:
            return False

    def nearest_node(self, xRand):
        vertexList = []
        for eachVertex in self.treeVertex:
            distX = xRand.x - eachVertex.x
            distY = xRand.y - eachVertex.y
            dist = np.linalg.norm([distX, distY])
            vertexList.append(dist)
        minIndex = np.argmin(vertexList)
        xNear = self.treeVertex[minIndex]
        return xNear

    def steer(self, xNearest, xRand):
        distX = xRand.x - xNearest.x
        distY = xRand.y - xNearest.y
        dist = np.linalg.norm([distX, distY])
        if dist <= self.eta:
            xNew = xRand
        else:
            direction = np.arctan2(distY, distX)
            newX = self.eta * np.cos(direction) + xNearest.x
            newY = self.eta * np.sin(direction) + xNearest.y
            xNew = Node(newX, newY)
        return xNew

    def near(self, xNew, minStep):
        neighbor = []
        for index, vertex in enumerate(self.treeVertex):
            dist = np.linalg.norm([(xNew.x - vertex.x), (xNew.y - vertex.y)])
            if dist <= minStep:
                neighbor.append(index)
        return [self.treeVertex[i] for i in neighbor]

    def cost_line(self, xStart, xEnd):
        return np.linalg.norm([(xStart.x - xEnd.x), (xStart.y - xEnd.y)])  # simple euclidean distance as cost

    def rotation_to_world_frame(self, xStart, xGoal):
        theta = np.arctan2((xGoal.y - xStart.y), (xGoal.x - xStart.x))

        R = np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]]).T

        return R

    def collision_check_node(self, xNew):
        nodepoint = ObjPoint2D(xNew.x, xNew.y)
        col = []
        for obs in self.obs:
            colide = intersect_point_v_rectangle(nodepoint, obs)
            col.append(colide)
        if True in col:
            return True
        else:
            return False

    def collision_check_line(self, xNearest, xNew):
        line = ObjLine2D(xNearest.x, xNearest.y, xNew.x, xNew.y)
        col = []
        for obs in self.obs:
            colide = intersect_line_v_rectangle(line, obs)
            col.append(colide)
        if True in col:
            return True
        else:
            return False

    def plot_env(self, after_plan=False):
        # plot obstacle
        for obs in self.obs:
            obs.plot()

        if after_plan:
            # plot tree vertex and branches
            for j in self.treeVertex:
                plt.scatter(j.x, j.y, color="red")  # vertex
                if j is not self.xStart:
                    plt.plot([j.x, j.parent.x], [j.y, j.parent.y], color="green")  # branch

        # plot start and goal node
        plt.scatter([self.xStart.x, self.xGoal.x], [self.xStart.y, self.xGoal.y], color='cyan')

        # plot ingoal region node
        for l in self.XSoln:
            plt.scatter(l.x, l.y, color="yellow")


if __name__ == "__main__":
    from map.taskmap_geo_format import task_rectangle_obs_7
    from map.taskmap_img_format import bmap
    from map.mapclass import CostMapLoader, CostMapClass, GeoMapClass
    np.random.seed(9)

    # SECTION - Experiment 1
    # start = np.array([4, 4]).reshape(2, 1)
    # goal = np.array([7, 8]).reshape(2, 1)
    # mapclass = GeoMapClass(geomap=task_rectangle_obs_7(), maprange=[[0, 10], [0, 10]])

    # SECTION - Experiment 2
    start = np.array([4, 4]).reshape(2, 1)
    goal = np.array([8.5, 1]).reshape(2, 1)
    maploader = CostMapLoader.loadarray(bmap())
    mapclass = CostMapClass(maploader, maprange=[[0, 10], [0, 10]])

    # SECTION - Planing Section
    planner = InfmRrtstar(mapclass=mapclass, xStart=start, xGoal=goal, maxIteration=500)
    planner.plot_env()
    plt.show()
    planner.planning()
    path = planner.search_path()

    # SECTION - plot planning result
    planner.plot_env(after_plan=True)
    plt.plot([node.x for node in path], [node.y for node in path], color='blue')
    plt.show()
