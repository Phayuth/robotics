""" Path Planning for Planar RR with Bi-directional RRT
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
        self.treeVertexStart = [self.xStart]
        self.treeVertexGoal = [self.xGoal]
        self.nearStart = None
        self.nearGoal = None

    def planning(self):
        for itera in range(self.maxIteration):
            print(itera)
            xRand = self.sampling()
            xNearestStart = self.nearest_node_tree_start(xRand)
            xNearestGoal = self.nearest_node_tree_goal(xRand)
            xNewStart = self.steer(xNearestStart, xRand)
            xNewGoal = self.steer(xNearestGoal, xRand)
            xNewStart.parent = xNearestStart
            xNewGoal.parent = xNearestGoal
            if self.collision_check_node(xNewStart) or self.collision_check_line(xNearestStart, xNewStart):
                continue
            else:
                self.treeVertexStart.append(xNewStart)

            if self.collision_check_node(xNewGoal) or self.collision_check_line(xNearestGoal, xNewGoal):
                continue
            else:
                self.treeVertexGoal.append(xNewGoal)

            if self.if_both_tree_node_near():
                break

    def search_path(self):
        pathStart = [self.nearStart]
        currentNode = self.nearStart
        while currentNode != self.xStart:
            currentNode = currentNode.parent
            pathStart.append(currentNode)

        pathGoal = [self.nearGoal]
        currentNode = self.nearGoal
        while currentNode != self.xGoal:
            currentNode = currentNode.parent
            pathStart.append(currentNode)

        # pathStart.reverse()

        # path = pathStart.extend(pathGoal)

        return pathStart, pathGoal

    def sampling(self):
        x = np.random.uniform(low=self.xMinRange, high=self.xMaxRange)
        y = np.random.uniform(low=self.yMinRange, high=self.yMaxRange)
        xRand = Node(x, y)

        return xRand

    def nearest_node_tree_start(self, xRand):
        vertexList = []

        for eachVertex in self.treeVertexStart:
            distX = xRand.x - eachVertex.x
            distY = xRand.y - eachVertex.y
            dist = np.linalg.norm([distX, distY])
            vertexList.append(dist)

        minIndex = np.argmin(vertexList)
        xNear = self.treeVertexStart[minIndex]

        return xNear
    
    def nearest_node_tree_goal(self, xRand):
        vertexList = []

        for eachVertex in self.treeVertexGoal:
            distX = xRand.x - eachVertex.x
            distY = xRand.y - eachVertex.y
            dist = np.linalg.norm([distX, distY])
            vertexList.append(dist)

        minIndex = np.argmin(vertexList)
        xNear = self.treeVertexGoal[minIndex]

        return xNear

    def if_both_tree_node_near(self):
        for eachVertexStart in self.treeVertexStart:
            for eachVertexGoal in self.treeVertexGoal:
                distX = eachVertexStart.x - eachVertexGoal.x
                distY = eachVertexStart.y - eachVertexGoal.y
                if np.linalg.norm([distX, distY]) < self.eta:
                    self.nearStart = eachVertexStart
                    self.nearGoal = eachVertexGoal
                    return True
        return False

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
        for obs in self.obs:
            obs.plot()

        if after_plan:
            for j in self.treeVertexStart:
                plt.scatter(j.x, j.y, color="red")
                if j is not self.xStart:
                    plt.plot([j.x, j.parent.x], [j.y, j.parent.y], color="green")

            for j in self.treeVertexGoal:
                plt.scatter(j.x, j.y, color="pink")
                if j is not self.xGoal:
                    plt.plot([j.x, j.parent.x], [j.y, j.parent.y], color="blueviolet")

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
    goal = np.array([0.9, 0.9]).reshape(2, 1)


    # SECTION - Experiment 3
    # start = np.array([4, 4]).reshape(2, 1)
    # goal = np.array([8.5, 1]).reshape(2, 1)
    # maploader = CostMapLoader.loadarray(bmap())
    # mapClass = CostMapClass(maploader, maprange=[[0, 10], [0, 10]])


    # SECTION - Planning Section
    planner = RRTBase(mapClass, start, goal, eta=0.1, maxIteration=100)
    planner.plot_env()
    plt.show()
    planner.planning()
    paths, pathg = planner.search_path()
    print(f"==>> paths: \n{paths}")
    print(f"==>> pathg: \n{pathg}")


    # SECTION - plot planning result
    planner.plot_env(after_plan=True)
    plt.plot([node.x for node in paths], [node.y for node in paths], color='blue')
    plt.plot([node.x for node in pathg], [node.y for node in pathg], color='blue')
    plt.show()
