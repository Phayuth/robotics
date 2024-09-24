import os
import sys

sys.path.append(str(os.path.abspath(os.getcwd())))

import time
import numpy as np
from planner.graph_based.graph import Node, save_graph, load_graph
from planner.graph_based.dijkstra import Dijkstra
from planner.graph_based.astar import AStar


class PRMComponent:

    def __init__(self, **kwargs):
        # simulator
        self.simulator = kwargs["simulator"]
        self.configLimit = np.array(self.simulator.configLimit)
        self.configDoF = self.simulator.configDoF

        # planner properties : general, some parameter must be set and some are optinal with default value
        self.eta = kwargs["eta"]
        self.subEta = kwargs["subEta"]
        self.discreteLimitNumSeg = kwargs.get("discreteLimitNumSeg", 10)  # limited number of segment divide for collision check and in goal region check

        # graph
        self.nodes = []

    def build_graph(self):
        raise NotImplementedError

    def save_graph(self, path):
        save_graph(self.nodes, path)

    def load_graph(self, path):
        self.nodes = load_graph(path)

    def query(self, xStart, xGoal, nodes, searcher="dij"):
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

    def query_multiple_goal(self, xStart, xGoals: list, nodes, searcher="dij"):
        xStart = Node(xStart)
        xGoals = [Node(xi) for xi in xGoals]

        # find nearest node of start and goal to nodes in roadmap
        # check if query nodes is too far or in collision
        xNearestStart = self.nearest_node(nodes, xStart)
        xNearestGoals = [self.nearest_node(nodes, xi) for xi in xGoals]

        # do the search between nearest start and goal
        if searcher == "dij":
            schr = Dijkstra(nodes)
        if searcher == "ast":
            schr = AStar(nodes)
        path = schr.search_multiple_goal(xNearestStart, xNearestGoals)
        path.reverse()
        xNearestGoalID = xNearestGoals.index(path[-1])

        # add start and goal to each end
        path.insert(0, xStart)
        path.append(xGoals[xNearestGoalID])

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

    def near(self, treeVertices, xCheck, searchRadius=None):
        if searchRadius is None:
            searchRadius = self.eta

        distListToxCheck = np.array([self.distance_between_config(xCheck, vertex) for vertex in treeVertices])
        nearIndices = np.where(distListToxCheck <= searchRadius)[0]
        return [treeVertices[item] for item in nearIndices], distListToxCheck[nearIndices]

    def steer(self, xFrom, xTo, distance, returnIsReached=False):
        distI = xTo.config - xFrom.config
        dist = np.linalg.norm(distI)
        isReached = False
        if dist <= distance:
            xNew = Node(xTo.config)
            isReached = True
        else:
            dI = (distI / dist) * distance
            newI = xFrom.config + dI
            xNew = Node(newI)
        if returnIsReached:
            return xNew, isReached
        else:
            return xNew

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


class PRMPlotter:

    def plot_2d_obstacle(simulator, axis):
        joint1Range = np.linspace(simulator.configLimit[0][0], simulator.configLimit[0][1], 360)
        joint2Range = np.linspace(simulator.configLimit[1][0], simulator.configLimit[1][1], 360)
        collisionPoint = []
        for theta1 in joint1Range:
            for theta2 in joint2Range:
                config = np.array([[theta1], [theta2]])
                result = simulator.collision_check(config)
                if result is True:
                    collisionPoint.append([theta1, theta2])

        collisionPoint = np.array(collisionPoint)
        axis.plot(collisionPoint[:, 0], collisionPoint[:, 1], color="darkcyan", linewidth=0, marker="o", markerfacecolor="darkcyan", markersize=1.5)

    def plot_2d_roadmap(nodes, axis):
        for ns in nodes:
            for nc in ns.edgeNodes:
                axis.plot([ns.config[0], nc.config[0]], [ns.config[1], nc.config[1]], color="darkgray")

    def plot_2d_state(xStart, xGoal, axis):
        axis.plot(xStart.config[0], xStart.config[1], color="blue", linewidth=0, marker="o", markerfacecolor="yellow")

        if isinstance(xGoal, list):
            for xG in xGoal:
                axis.plot(xG.config[0], xG.config[1], color="blue", linewidth=0, marker="o", markerfacecolor="red")
        else:
            axis.plot(xGoal.config[0], xGoal.config[1], color="blue", linewidth=0, marker="o", markerfacecolor="red")

    def plot_2d_path(path, axis):
        axis.plot([p.config[0] for p in path], [p.config[1] for p in path], color="blue", linewidth=2, marker="o", markerfacecolor="plum", markersize=5)

    def plot_2d_complete(path=None, plannerClass=None, ax=None):
        PRMPlotter.plot_2d_obstacle(plannerClass.simulator, ax)
        PRMPlotter.plot_2d_roadmap(plannerClass.nodes, ax)
        if path:
            PRMPlotter.plot_2d_path(path, ax)
            PRMPlotter.plot_2d_state(path[0], path[-1], ax)

    def plot_performance():
        pass