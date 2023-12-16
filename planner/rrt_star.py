import os
import sys

wd = os.path.abspath(os.getcwd())
sys.path.append(str(wd))

import numpy as np
from planner.rrt_component import Node, RRTComponent


class RRTStar(RRTComponent):

    def __init__(self, xStart, xApp, xGoal, config):
        super().__init__(config)
        # start, aux, goal node
        self.xStart = Node(xStart)
        self.xGoal = Node(xGoal)
        self.xApp = Node(xApp)

        # planner properties
        self.treeVertex = [self.xStart]
        self.distGoalToApp = self.distance_between_config(self.xGoal, self.xApp)

        # local sampling properties
        self.localOptEnable = config["localOptEnable"]
        if self.localOptEnable:
            self.anchorPath = None
            self.localPath = None
            self.numSegSamplingNode = None

        # solutions
        self.XInGoalRegion = []

    @RRTComponent.catch_key_interrupt
    def start(self):
        for itera in range(self.maxIteration):
            self.cBestNow = self.cbest_single_tree(self.XInGoalRegion, self.xApp, itera)  # save cost graph

            if self.localOptEnable:
                if len(self.XInGoalRegion) == 0:
                    xRand = self.bias_uniform_sampling(self.xApp, len(self.XInGoalRegion))
                else:
                    xRand = self.local_path_sampling(self.anchorPath, self.localPath, self.numSegSamplingNode)
            else:
                xRand = self.bias_uniform_sampling(self.xApp, len(self.XInGoalRegion))

            xNearest, vertexDistList = self.nearest_node(self.treeVertex, xRand, returnDistList=True)
            xNew, xNewIsxRand = self.steer(xNearest, xRand, self.eta, returnIsReached=True)
            if self.is_collision_and_in_goal_region(xNearest, xNew, self.xGoal, self.distGoalToApp):
                continue
            xNew.parent = xNearest
            xNew.cost = xNew.parent.cost + self.cost_line(xNew, xNew.parent)
            xNearest.child.append(xNew)

            # rrtstar
            self.star_optimizer(self.treeVertex, xNew, self.rewireRadius, xNewIsxRand, vertexDistList)

            # in goal region
            if self.is_config_in_region_of_config(xNew, self.xApp, radius=self.nearGoalRadius):
                self.XInGoalRegion.append(xNew)
                if self.localOptEnable:
                    if self.anchorPath is None:
                        self.localPath = self.search_backtrack_single_directional_path(self.XInGoalRegion[0], self.xApp)
                        self.numSegSamplingNode = len(self.localPath) - 1
                        self.anchorPath = self.segment_interpolation_between_config(self.xStart, self.xApp, self.numSegSamplingNode, includexStart=True)

            if self.termination_check(self.XInGoalRegion):
                break

    def get_path(self):
        return self.search_best_cost_singledirection_path(backFromNode=self.xApp, treeVertexList=self.XInGoalRegion, attachNode=self.xGoal)

    def update_perf(self, timePlanningStart, timePlanningEnd):
        self.perf_matrix_update(tree1=self.treeVertex, tree2=None, timePlanningStart=timePlanningStart, timePlanningEnd=timePlanningEnd)


class RRTStarMulti(RRTComponent):

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
        self.XInGoalRegion = [[] for _ in range(self.numGoal)]
        self.xGoalBestIndex = None

    @RRTComponent.catch_key_interrupt
    def start(self):
        for itera in range(self.maxIteration):
            self.cBestNow, self.xGoalBestIndex = self.cbest_single_tree_multi(self.XInGoalRegion, self.xAppList, itera)

            biasIndex = np.random.randint(low=0, high=self.numGoal)
            xRand = self.bias_uniform_sampling(self.xAppList[biasIndex], len(self.XInGoalRegion[biasIndex]))
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
                    self.XInGoalRegion[id].append(xNew)

            if self.termination_check(self.XInGoalRegion):
                break

    def get_path(self):
        return self.search_best_cost_singledirection_path(backFromNode=self.xAppList[self.xGoalBestIndex],
                                                          treeVertexList=self.XInGoalRegion[self.xGoalBestIndex],
                                                          attachNode=self.xGoalList[self.xGoalBestIndex])

    def update_perf(self, timePlanningStart, timePlanningEnd):
        self.perf_matrix_update(tree1=self.treeVertex, tree2=None, timePlanningStart=timePlanningStart, timePlanningEnd=timePlanningEnd)