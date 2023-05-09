""" Path Planning with RRT Star with reject sampling in goal region
- Map : create from image to geometry with MapLoader and MapClass
- Collision : geometry check
- Searcher : Cost search
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


class RRTStar:

    def __init__(self, mapclass, xStart, xGoal, eta=None, maxIteration=1000) -> None:
        # map properties
        self.mapclass = mapclass
        self.xMinRange = self.mapclass.xmin
        self.xMaxRange = self.mapclass.xmax
        self.yMinRange = self.mapclass.ymin
        self.yMaxRange = self.mapclass.ymax
        self.xStart = Node(xStart[0, 0], xStart[1, 0])
        self.xStart.cost = 0.0
        self.xGoal = Node(xGoal[0, 0], xGoal[1, 0])

        if mapclass.__class__.__name__ == "CostMapClass":
            self.obs = self.mapclass.costmap2geo()
        else:
            self.obs = self.mapclass.obj

        # properties of planner
        self.maxIteration = maxIteration
        self.m = (self.xMaxRange - self.xMinRange) * (self.yMaxRange - self.yMinRange)
        self.radius = (2 * (1 + 1/2)**(1 / 2)) * (self.m / np.pi)**(1 / 2)
        self.eta = self.radius * (np.log(self.maxIteration) / self.maxIteration)**(1 / 2)
        self.treeVertex = [self.xStart]

    def planning(self):
        for itera in range(self.maxIteration):
            print(itera)
            xRand = self.sampling()
            xNearest = self.nearest_node(xRand)
            xNew = self.steer(xNearest, xRand)
            if self.node_in_goal_region(xNew) or self.connect_in_goal_region(xNearest, xNew):
                continue
            xNew.parent = xNearest
            xNew.cost = xNew.parent.cost + self.cost_line(xNew, xNew.parent)
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

    def search_path(self):
        XNear = self.near(self.xGoal, self.eta)
        for xNear in XNear:
            if self.collision_check_line(xNear, self.xGoal):
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

    def sampling(self):
        x = np.random.uniform(low=self.xMinRange, high=self.xMaxRange)
        y = np.random.uniform(low=self.yMinRange, high=self.yMaxRange)
        xRand = Node(x, y)
        return xRand

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

    def cost_line(self, xstart, xend):
        return np.linalg.norm([(xstart.x - xend.x), (xstart.y - xend.y)])

    def node_in_goal_region(self, xNew):
        distX = self.xGoal.x - xNew.x
        distY = self.xGoal.y - xNew.y
        dist = np.linalg.norm([distX, distY])

        if dist <= 0.5:
            return True
        else:
            return False
        
    def connect_in_goal_region(self, xNearest, xNew):
        distX = xNew.x - xNearest.x
        distY = xNew.y - xNearest.y
        desiredStep = 10
        rateX = distX / desiredStep
        rateY = distY / desiredStep

        for i in range(1, desiredStep - 1):
            newX = xNearest.x + (rateX * i)
            newY = xNearest.y + (rateY * i)
            xNew = Node(newX, newY)
            if self.node_in_goal_region(xNew):
                return True
        return False
    
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
                plt.scatter(j.x, j.y, color="lightyellow")  # vertex
                if j is not self.xStart:
                    plt.plot([j.x, j.parent.x], [j.y, j.parent.y], color="skyblue")  # branch

        # plot start and goal Node
        plt.scatter([self.xStart.x, self.xGoal.x], [self.xStart.y, self.xGoal.y], color='gold')


if __name__ == "__main__":
    from map.taskmap_geo_format import task_rectangle_obs_7
    from map.taskmap_img_format import bmap
    from map.mapclass import CostMapLoader, CostMapClass, GeoMapClass
    np.random.seed(9)


    # SECTION - Experiment 1
    mapClass = GeoMapClass(geomap=task_rectangle_obs_7(), maprange=[[0, 10], [0, 10]])
    start = np.array([4,4]).reshape(2,1)
    goal = np.array([7,8]).reshape(2,1)


    # SECTION - Experiment 2
    # mapLoader = CostMapLoader.loadarray(bmap())
    # mapClass = CostMapClass(maploader=mapLoader, maprange=[[-np.pi, np.pi], [-np.pi, np.pi]])
    # start = np.array([0, 0]).reshape(2, 1)
    # goal = np.array([1, 1]).reshape(2, 1)


    # SECTION - planning section
    planner = RRTStar(mapClass, start, goal, maxIteration=3000)
    planner.plot_env()
    plt.show()
    planner.planning()
    path = planner.search_path()


    # SECTION - plot result
    planner.plot_env(after_plan=True)
    plt.plot([node.x for node in path], [node.y for node in path], color='blue')
    plt.show()
