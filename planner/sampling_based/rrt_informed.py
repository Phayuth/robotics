import os
import sys

sys.path.append(str(os.path.abspath(os.getcwd())))

import numpy as np
from planner.sampling_based.rrt_component import Node, RRTComponent


class RRTInformed(RRTComponent):

    def __init__(self, xStart, xApp, xGoal, config) -> None:
        super().__init__(config)
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
            self.cBestNow = self.cbest_single_tree(self.XSoln, self.xApp, itera)

            if self.cBestNow == np.inf:
                xRand = self.bias_uniform_sampling(self.xApp, len(self.XSoln))
            elif self.cBestNow < np.inf:
                xRand = self.informed_sampling(self.xCenter, self.cBestNow, self.cMin, self.C)

            xNearest, vertexDistList = self.nearest_node(self.treeVertex, xRand, returnDistList=True)
            xNew, xNewIsxRand = self.steer(xNearest, xRand, self.eta, returnIsReached=True)
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

    def get_path(self):
        return self.search_best_cost_singledirection_path(backFromNode=self.xApp, treeVertexList=self.XSoln, attachNode=self.xGoal)


class RRTInformedMulti(RRTComponent):

    def __init__(self, xStart, xAppList, xGoalList, config):
        super().__init__(config)
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
            self.cBestNow, self.xGoalBestIndex = self.cbest_single_tree_multi(self.XSoln, self.xAppList, itera)

            if self.cBestNow == np.inf:
                xRand = self.bias_uniform_sampling(self.xAppList[biasIndex], len(self.XSoln[biasIndex]))
            elif self.cBestNow < np.inf:
                xRand = self.informed_sampling(self.xCenter[biasIndex], self.cBestNow, self.cMin[biasIndex], self.C[biasIndex])

            xNearest, vertexDistList = self.nearest_node(self.treeVertex, xRand, returnDistList=True)
            xNew, xNewIsxRand = self.steer(xNearest, xRand, self.eta, returnIsReached=True)
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

    def get_path(self):
        return self.search_best_cost_singledirection_path(backFromNode=self.xAppList[self.xGoalBestIndex],
                                                          treeVertexList=self.XSoln[self.xGoalBestIndex],
                                                          attachNode=self.xGoalList[self.xGoalBestIndex])