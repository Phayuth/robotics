import os
import sys

wd = os.path.abspath(os.getcwd())
sys.path.append(str(wd))

import time
import numpy as np
from planner.graph_based.dijkstra import Dijkstra
from planner.graph_based.astar import AStar


class Node:

    def __init__(self, config=None, name=None) -> None:
        self.config = config
        self.name = name
        self.cost = np.inf
        self.pathvia = None
        self.edgeNodes = []
        self.edgeCosts = []

    def __repr__(self) -> str:
        return f"config : {self.config.T}, #edges : {len(self.edgeNodes)}"


class Edge:

    def __init__(self, nodeA: Node, nodeB: Node) -> None:
        self.nodeA = nodeA
        self.nodeB = nodeB
        self.cost = None

    def __repr__(self) -> str:
        return f"AConfig = {self.nodeA.config.T}, BConfig = {self.nodeB.config.T}, Cost = {self.cost}"


class PRMComponent:

    def __init__(self, **kwargs):
        # simulator
        self.simulator = kwargs["simulator"]
        self.configLimit = np.array(self.simulator.configLimit)
        self.configDoF = self.simulator.configDoF

        # planner properties : general, some parameter must be set and some are optinal with default value
        self.eta = kwargs["eta"]
        self.subEta = kwargs["subEta"]
        self.kNNTopNearest = kwargs.get("kNNTopNearest", 10)  # search all neighbour but only return top nearest nighbour during near neighbour search, if None, return all
        self.discreteLimitNumSeg = kwargs.get("discreteLimitNumSeg", 10)  # limited number of segment divide for collision check and in goal region check

    def build_roadmap(self):
        raise NotImplementedError

    def save_roadmap(self):
        pass

    def load_roadmap(self):
        pass

    def query(self, xStart: Node, xGoal: Node, nodes, searcher="dij"):
        xStart = Node(xStart)
        xGoal = Node(xGoal)

        # find nearest node of start and goal to nodes in roadmap
        # check if query nodes is too far or in collision
        xNearestStart = self.nearest_node(nodes, xStart)
        xNearestGoal = self.nearest_node(nodes, xGoal)

        # do the search between nearest start and goal
        if searcher == "dij":
            schr = Dijkstra(nodes)
        if searcher == "ast":
            schr = AStar(nodes)
        path = schr.search(xNearestStart, xNearestGoal)
        path.reverse()

        # add start and goal to each end
        path.insert(0, xStart)
        path.append(xGoal)

        return path

    def uni_sampling(self) -> Node:
        config = np.random.uniform(low=self.configLimit[:, 0], high=self.configLimit[:, 1]).reshape(-1, 1)
        xRand = Node(config)
        return xRand

    def nearest_node(self, treeVertices, xCheck, returnDistList=False):
        distListToxCheck = [self.distance_between_config(xCheck, x) for x in treeVertices]
        minIndex = np.argmin(distListToxCheck)
        xNearest = treeVertices[minIndex]
        if returnDistList:
            return xNearest, distListToxCheck
        else:
            return xNearest

    def near(self, treeVertices, xCheck, searchRadius=None, distListToxCheck=None):
        if searchRadius is None:
            searchRadius = self.eta

        if distListToxCheck:
            distListToxCheck = np.array(distListToxCheck)
        else:
            distListToxCheck = np.array([self.distance_between_config(xCheck, vertex) for vertex in treeVertices])

        nearIndices = np.where(distListToxCheck <= searchRadius)[0]

        if self.kNNTopNearest:
            if len(nearIndices) < self.kNNTopNearest:
                return [treeVertices[item] for item in nearIndices], distListToxCheck[nearIndices]
            else:
                nearDistList = distListToxCheck[nearIndices]
                sortedIndicesDist = np.argsort(nearDistList)
                topNearIndices = nearIndices[sortedIndicesDist[: self.kNNTopNearest]]
                return [treeVertices[item] for item in topNearIndices], distListToxCheck[topNearIndices]
        else:
            return [treeVertices[item] for item in nearIndices], distListToxCheck[nearIndices]

    def cost_line(self, xFrom, xTo):
        return self.distance_between_config(xFrom, xTo)

    def is_collision(self, xFrom, xTo):
        if self.is_config_in_collision(xTo):
            return True
        elif self.is_connect_config_in_collision(xFrom, xTo):
            return True
        else:
            return False

    def is_config_in_collision(self, xCheck):
        result = self.simulator.collision_check(xCheck.config)
        return result

    def is_connect_config_in_collision(self, xFrom, xTo, NumSeg=None):
        distI = self.distance_each_component_between_config(xFrom, xTo)
        dist = np.linalg.norm(distI)
        if NumSeg:
            NumSeg = NumSeg
        else:
            NumSeg = int(np.ceil(dist / self.subEta))
            if NumSeg > self.discreteLimitNumSeg:
                NumSeg = self.discreteLimitNumSeg
        rateI = distI / NumSeg
        for i in range(1, NumSeg):
            newI = xFrom.config + rateI * i
            xTo = Node(newI)
            if self.is_config_in_collision(xTo):
                return True
        return False

    def distance_between_config(self, xFrom, xTo):
        return np.linalg.norm(self.distance_each_component_between_config(xFrom, xTo))

    def distance_each_component_between_config(self, xFrom, xTo):
        return xTo.config - xFrom.config
