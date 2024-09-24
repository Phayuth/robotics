import os
import sys

sys.path.append(str(os.path.abspath(os.getcwd())))

import time
import numpy as np
from planner.graph_based.graph import Node, save_graph, load_graph
from planner.graph_based.dijkstra import Dijkstra
from planner.graph_based.astar import AStar
from spatial_geometry.utils import Utils


class PRMTorusRedundantComponent:

    def __init__(self, **kwargs):
        # simulator
        self.simulator = kwargs["simulator"]
        self.configLimit = np.array(self.simulator.configLimit)  # should be -+ pi range
        self.configDoF = self.simulator.configDoF
        self.lim = np.array([[-2 * np.pi, 2 * np.pi] * self.configDoF])

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
        xNearestStart = self.nearest_wrap_node(nodes, xStart)
        xNearestGoal = self.nearest_wrap_node(nodes, xGoal)

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

    def grid_sampling(self, nodeperdimension=25):
        x = np.linspace(-np.pi, np.pi, nodeperdimension)
        y = np.linspace(-np.pi, np.pi, nodeperdimension)

        X, Y = np.meshgrid(x, y)

        data = []
        for i, j in np.ndindex(X.shape):
            data.append([X[i, j], Y[i, i]])
        data = np.array(data)
        XRand = [Node(data[i].reshape(-1, 1)) for i in range(data.shape[0])]
        return XRand

    def nearest_wrap_node(self, treeVertices: list[Node], xCheck: Node):
        dist = [Utils.minimum_dist_torus(xCheck.config, xi.config) for xi in treeVertices]
        minIndex = np.argmin(dist)
        return treeVertices[minIndex]

    def near_wrap(self, treeVertices, xCheck, searchRadius=None):
        if searchRadius is None:
            searchRadius = self.eta

        distListToxCheck = np.array([Utils.minimum_dist_torus(xCheck.config, vertex.config) for vertex in treeVertices])
        nearIndices = np.where(distListToxCheck <= searchRadius)[0]
        return [treeVertices[item] for item in nearIndices], distListToxCheck[nearIndices]

    def cost_line(self, xFrom, xTo):
        return Utils.minimum_dist_torus(xFrom.config, xTo.config)

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
        distI = Utils.nearest_qb_to_qa(xFrom.config, xTo.config, self.lim, ignoreOrginal=False) - xFrom.config
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


class PRMTorusRedundantPlotter:

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
        limt2 = np.array([[-2 * np.pi, 2 * np.pi], [-2 * np.pi, 2 * np.pi]])
        for ns in nodes:
            for nc in ns.edgeNodes:
                qabw = Utils.nearest_qb_to_qa(ns.config, nc.config, limt2, ignoreOrginal=False)
                qbaw = Utils.nearest_qb_to_qa(nc.config, ns.config, limt2, ignoreOrginal=False)
                axis.plot([ns.config[0], qabw[0]], [ns.config[1], qabw[1]], color="darkgray", marker="o", markerfacecolor="black")
                axis.plot([nc.config[0], qbaw[0]], [nc.config[1], qbaw[1]], color="darkgray", marker="o", markerfacecolor="black")

    def plot_2d_state(xStart, xGoal, axis):
        axis.plot(xStart.config[0], xStart.config[1], color="blue", linewidth=0, marker="o", markerfacecolor="yellow")

        if isinstance(xGoal, list):
            for xG in xGoal:
                axis.plot(xG.config[0], xG.config[1], color="blue", linewidth=0, marker="o", markerfacecolor="red")
        else:
            axis.plot(xGoal.config[0], xGoal.config[1], color="blue", linewidth=0, marker="o", markerfacecolor="red")

    def plot_2d_path(path, axis):
        limt2 = np.array([[-2 * np.pi, 2 * np.pi], [-2 * np.pi, 2 * np.pi]])
        pc = path[0].config
        for i in range(1, len(path)):
            qabw = Utils.nearest_qb_to_qa(pc, path[i].config, limt2, ignoreOrginal=False)
            qbaw = Utils.nearest_qb_to_qa(path[i].config, pc, limt2, ignoreOrginal=False)
            axis.plot([pc[0], qabw[0]], [pc[1], qabw[1]], color="blue", linewidth=2, marker="o", markerfacecolor="plum", markersize=5)
            axis.plot([path[i].config[0], qbaw[0]], [path[i].config[1], qbaw[1]], color="blue", linewidth=2, marker="o", markerfacecolor="plum", markersize=5)
            pc = path[i].config

    def plot_2d_complete(path=None, plannerClass=None, ax=None):
        PRMTorusRedundantPlotter.plot_2d_obstacle(plannerClass.simulator, ax)
        PRMTorusRedundantPlotter.plot_2d_roadmap(plannerClass.nodes, ax)
        if path:
            PRMTorusRedundantPlotter.plot_2d_path(path, ax)
            PRMTorusRedundantPlotter.plot_2d_state(path[0], path[-1], ax)

    def plot_performance():
        pass