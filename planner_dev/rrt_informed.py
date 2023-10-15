import os
import sys

wd = os.path.abspath(os.getcwd())
sys.path.append(str(wd))

import numpy as np
from planner_dev.rrt_component import Node, RRTComponent


class RRTInformed(RRTComponent):

    def __init__(self, xStart, xApp, xGoal, eta, subEta, maxIteration, numDoF, envChoice, nearGoalRadius, rewireRadius, terminationConditionID, print_debug) -> None:
        super().__init__(eta=eta,
                         subEta=subEta,
                         maxIteration=maxIteration,
                         numDoF=numDoF,
                         envChoice=envChoice,
                         nearGoalRadius=nearGoalRadius,
                         rewireRadius=rewireRadius,
                         terminationConditionID=terminationConditionID,
                         print_debug=print_debug)
        # start, aux, goal node
        self.xStart = Node(xStart)
        self.xGoal = Node(xGoal)
        self.xApp = Node(xApp)

        # planner properties
        self.treeVertex = [self.xStart]
        self.distGoalToApp = self.distance_between_config(self.xGoal, self.xApp)

        # solutions
        self.XSoln = []

        # informed sampling properties
        self.C = self.rotation_to_world(self.xStart, self.xApp)  # hyperellipsoid rotation axis
        self.cMin = self.distance_between_config(self.xStart, self.xApp)
        self.xCenter = (self.xStart.config + self.xApp.config) / 2

    @RRTComponent.catch_key_interrupt
    def start(self):
        for itera in range(self.maxIteration):
            self.cBestNow = self.cbest_single_tree(self.XSoln, self.xApp, itera, self.print_debug)

            if self.cBestNow == np.inf:
                xRand = self.bias_uniform_sampling(self.xApp, len(self.XSoln))
            elif self.cBestNow < np.inf:
                xRand = self.informed_sampling(self.xCenter, self.cBestNow, self.cMin, self.C)

            xNearest, vertexDistList = self.nearest_node(self.treeVertex, xRand, returnDistList=True)
            xNew, xNewIsxRand = self.steer(xNearest, xRand, self.eta, returnxNewIsxRand=True)
            if self.is_collision_and_in_goal_region(xNearest, xNew, self.xGoal, self.distGoalToApp):
                continue
            xNew.parent = xNearest
            xNew.cost = xNew.parent.cost + self.cost_line(xNew, xNew.parent)
            xNearest.child.append(xNew)

            self.star_optimizer(self.treeVertex, xNew, self.rewireRadius, xNewIsxRand, vertexDistList)

            # in approach region
            if self.is_config_in_region_of_config(xNew, self.xApp, radius=self.nearGoalRadius):
                self.XSoln.append(xNew)

            if self.termination_check(self.XSoln):
                break

    def plot_tree(self, path, ax):
        self.plot_2d_obstacle(ax)
        self.plot_2d_single_tree(self.treeVertex, ax)
        self.plot_2d_path(path, ax)
        self.plot_2d_state_configuration(self.xStart, self.xApp, self.xGoal, ax)


class RRTInformedMulti(RRTComponent):

    def __init__(self, xStart, xAppList, xGoalList, eta, subEta, maxIteration, numDoF, envChoice, nearGoalRadius, rewireRadius, terminationConditionID, print_debug):
        super().__init__(eta=eta,
                         subEta=subEta,
                         maxIteration=maxIteration,
                         numDoF=numDoF,
                         envChoice=envChoice,
                         nearGoalRadius=nearGoalRadius,
                         rewireRadius=rewireRadius,
                         terminationConditionID=terminationConditionID,
                         print_debug=print_debug)
        # start, aux, goal node
        self.xStart = Node(xStart)
        self.xGoalList = [Node(xGoali) for xGoali in xGoalList]
        self.xAppList = [Node(xAppi) for xAppi in xAppList]
        self.numGoal = len(self.xAppList)

        # planner properties
        self.treeVertex = [self.xStart]
        self.distGoalToAppList = [self.distance_between_config(xAppi, xGoali) for xAppi, xGoali in zip(self.xAppList, self.xGoalList)]

        # solutions
        self.XSoln = [[] for _ in range(self.numGoal)]
        self.xGoalBestIndex = None

        # informed sampling properties
        self.C = [self.rotation_to_world(self.xStart, xAppi) for xAppi in self.xAppList]
        self.cMin = [self.distance_between_config(self.xStart, xAppi) for xAppi in self.xAppList]
        self.xCenter = [(self.xStart.config + xAppi.config) / 2 for xAppi in self.xAppList]

    @RRTComponent.catch_key_interrupt
    def start(self):
        for itera in range(self.maxIteration):
            biasIndex = np.random.randint(low=0, high=self.numGoal)
            self.cBestNow, self.xGoalBestIndex = self.cbest_single_tree_multi(self.XSoln, self.xAppList, itera, self.print_debug)

            if self.cBestNow == np.inf:
                xRand = self.bias_uniform_sampling(self.xAppList[biasIndex], len(self.XSoln[biasIndex]))
            elif self.cBestNow < np.inf:
                xRand = self.informed_sampling(self.xCenter[biasIndex], self.cBestNow, self.cMin[biasIndex], self.C[biasIndex])

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
                    self.XSoln[id].append(xNew)

            if self.termination_check(self.XSoln):
                break

    def plot_tree(self, path, ax):
        self.plot_2d_obstacle(ax)
        self.plot_2d_single_tree(self.treeVertex, ax)
        self.plot_2d_path(path, ax)
        self.plot_2d_state_configuration(self.xStart, self.xAppList, self.xGoalList, ax)