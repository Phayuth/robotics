import os
import sys

wd = os.path.abspath(os.getcwd())
sys.path.append(str(wd))

import numpy as np
from planner_dev.rrt_component import Node, RRTComponent


class RRTStar(RRTComponent):

    def __init__(self, xStart, xApp, xGoal, eta, subEta, maxIteration, numDoF, envChoice, nearGoalRadius):
        super().__init__(NumDoF=numDoF, EnvChoice=envChoice)
        # start, aux, goal node
        self.xStart = Node(xStart)
        self.xGoal = Node(xGoal)
        self.xApp = Node(xApp)

        self.eta = eta
        self.subEta = subEta
        self.nearGoalRadius = nearGoalRadius
        self.maxIteration = maxIteration
        self.treeVertex = [self.xStart]
        self.distGoalToApp = self.distance_between_config(self.xGoal, self.xApp)
        self.rewireRadius = None
        self.XInGoalRegion = []

    @RRTComponent.catch_key_interrupt
    def start(self):
        for itera in range(self.maxIteration):
            print(itera)
            _ = self.single_tree_cbest(self.XInGoalRegion, self.xApp, itera)  # save cost graph

            xRand = self.bias_uniform_sampling(self.xApp, len(self.XInGoalRegion))
            xNearest, vertexDistList = self.nearest_node(self.treeVertex, xRand, returnDistList=True)
            xNew, xNewIsxRand = self.steer(xNearest, xRand, self.eta, returnxNewIsxRand=True)
            if self.is_collision_and_in_goal_region(xNearest, xNew, self.xGoal, self.distGoalToApp):
                continue
            xNew.parent = xNearest
            xNew.cost = xNew.parent.cost + self.cost_line(xNew, xNew.parent)
            xNearest.child.append(xNew)

            # rrtstar
            self.star_optimizer(self.treeVertex, xNew, self.rewireRadius, xNewIsxRand, vertexDistList)

            # in goal region,
            if self.is_config_in_region_of_config(xNew, self.xApp, radius=self.nearGoalRadius):
                self.XInGoalRegion.append(xNew)

    def plot_tree(self, path, ax):
        self.plot_2d_obstacle(ax)
        self.plot_2d_single_tree(self.treeVertex, ax)
        self.plot_2d_path(path, ax)
        self.plot_2d_state_configuration(self.xStart, self.xApp, self.xGoal, ax)


class RRTStarMulti(RRTComponent):

    def __init__(self, xStart, xAppList, xGoalList, eta, subEta, maxIteration, numDoF, envChoice, nearGoalRadius):
        super().__init__(NumDoF=numDoF, EnvChoice=envChoice)
        # start, aux, goal node
        self.xStart = Node(xStart)
        self.xGoalList = [Node(xGoali) for xGoali in xGoalList]
        self.xAppList = [Node(xAppi) for xAppi in xAppList]
        self.numGoal = len(self.xAppList)

        self.eta = eta
        self.subEta = subEta
        self.nearGoalRadius = nearGoalRadius
        self.maxIteration = maxIteration
        self.treeVertex = [self.xStart]
        self.distGoalToAppList = [self.distance_between_config(xAppi, xGoali) for xAppi, xGoali in zip(self.xAppList, self.xGoalList)]
        self.rewireRadius = None
        self.XInGoalRegion = [[] for _ in range(self.numGoal)]
        self.xGoalBestIndex = None

    @RRTComponent.catch_key_interrupt
    def start(self):
        for itera in range(self.maxIteration):
            print(itera)
            cBest, self.xGoalBestIndex = self.single_tree_multi_cbest(self.XInGoalRegion, self.xAppList, itera)

            biasIndex = np.random.randint(low=0, high=self.numGoal)
            xRand = self.bias_uniform_sampling(self.xAppList[biasIndex], len(self.XInGoalRegion[biasIndex]))
            xNearest, vertexDistList = self.nearest_node(self.treeVertex, xRand, returnDistList=True)
            xNew, xNewIsxRand = self.steer(xNearest, xRand, self.eta, returnxNewIsxRand=True)
            if self.is_collision(xNearest, xNew):
                continue
            xNew.parent = xNearest
            xNew.cost = xNew.parent.cost + self.cost_line(xNew, xNew.parent)
            xNearest.child.append(xNew)

            self.star_optimizer(self.treeVertex, xNew, self.rewireRadius, xNewIsxRand, vertexDistList)

            # in goal region
            for id, xApp in enumerate(self.xAppList):
                if self.is_config_in_region_of_config(xNew, xApp, radius=self.nearGoalRadius):
                    self.XInGoalRegion[id].append(xNew)

    def plot_tree(self, path, ax):
        self.plot_2d_obstacle(ax)
        self.plot_2d_single_tree(self.treeVertex, ax)
        self.plot_2d_path(path, ax)
        self.plot_2d_state_configuration(self.xStart, self.xAppList, self.xGoalList, ax)