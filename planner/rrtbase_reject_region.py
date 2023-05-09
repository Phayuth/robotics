""" Path Planning for Planar RR with RRT based with reject region near the goal
- Map : Continuous configuration space 2D create from image to geometry with MapLoader and MapClass
- Collsion : Geometry based
- Path Searcher : Naive Seach.

"""

import os
import sys

wd = os.path.abspath(os.getcwd())
sys.path.append(str(wd))

import numpy as np
import matplotlib.pyplot as plt
from collision_check_geometry.collision_class import ObjLine2D, ObjPoint2D, intersect_point_v_rectangle, intersect_line_v_rectangle

class Node:

    def __init__(self, x, y, parent=None) -> None:
        self.x = x
        self.y = y
        self.parent = parent


class RRTBase():

    def __init__(self, mapClass, xStart, xGoal, eta=0.3, maxIteration=1000) -> None:
        # precheck input data
        assert xStart.shape == (2,1), f"xStart must be numpy array"
        assert xGoal.shape == (2,1), f"xGoal must be numpy array"
        assert eta >= 0, f"eta must be positive"
        assert maxIteration >= 0, f"maxiteration "

        # map properties
        self.mapClass = mapClass
        self.xMinRange = self.mapClass.xmin
        self.xMaxRange = self.mapClass.xmax
        self.yMinRange = self.mapClass.ymin
        self.yMaxRange = self.mapClass.ymax
        self.xStart = Node(xStart[0, 0], xStart[1, 0])
        self.xGoal = Node(xGoal[0, 0], xGoal[1, 0])

        if mapClass.__class__.__name__ == "CostMapClass":
            self.obs = self.mapClass.costmap2geo()
        else:
            self.obs = self.mapClass.obj

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
            if self.node_in_goal_region(xNew) or self.connect_in_goal_region(xNearest, xNew):
                continue
            xNew.parent = xNearest
            if self.collision_check_node(xNew) or self.collision_check_line(xNearest, xNew):
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
                plt.scatter(j.x, j.y, color="red")
                if j is not self.xStart:
                    plt.plot([j.x, j.parent.x], [j.y, j.parent.y], color="green")

        # plot start and goal Node
        plt.scatter([self.xStart.x, self.xGoal.x], [self.xStart.y, self.xGoal.y], color='cyan')


if __name__ == "__main__":
    from map.taskmap_geo_format import task_rectangle_obs_3
    from map.taskmap_img_format import bmap
    from map.mapclass import CostMapLoader, CostMapClass, GeoMapClass
    np.random.seed(9)


    # SECTION - Experiment 1
    # maploader = CostMapLoader.loadarray(bmap())
    # mapClass = CostMapClass(maploader=maploader, maprange=[[-np.pi, np.pi], [-np.pi, np.pi]])
    # start = np.array([0, 0]).reshape(2, 1)
    # goal = np.array([1, 1]).reshape(2, 1)


    # SECTION - Experiment 2
    mapClass = GeoMapClass(geomap=task_rectangle_obs_3(), maprange=[[-np.pi, np.pi], [-np.pi, np.pi]])
    start = np.array([0, 0]).reshape(2, 1)
    goal = np.array([-2, -0.5]).reshape(2, 1)


    # SECTION - Experiment 3
    # start = np.array([4, 4]).reshape(2, 1)
    # goal = np.array([8.5, 1]).reshape(2, 1)
    # maploader = CostMapLoader.loadarray(bmap())
    # mapClass = CostMapClass(maploader, maprange=[[0, 10], [0, 10]])


    # SECTION - Planning Section
    planner = RRTBase(mapClass, start, goal, eta=0.1, maxIteration=3000)
    planner.plot_env()
    plt.show()
    planner.planning()
    path = planner.search_path()


    # SECTION - plot planning result
    planner.plot_env(after_plan=True)
    plt.plot([node.x for node in path], [node.y for node in path], color='blue')
    plt.show()
