import os
import sys

wd = os.path.abspath(os.getcwd())
sys.path.append(str(wd))

import numpy as np
from planner.rrt_component import Node, RRTComponent


class RRTBase(RRTComponent):

    def __init__(self, xStart, xApp, xGoal, eta, subEta, maxIteration, numDoF, envChoice, nearGoalRadius, rewireRadius, endIterationID, print_debug):
        super().__init__(eta, subEta, maxIteration, numDoF, envChoice, nearGoalRadius, rewireRadius, endIterationID, print_debug)
        # start, aux, goal node
        self.xStart = Node(xStart)
        self.xGoal = Node(xGoal)
        self.xApp = Node(xApp)

        # planner properties
        self.treeVertex = [self.xStart]
        self.distGoalToApp = self.distance_between_config(self.xGoal, self.xApp)

        # solutions
        self.XInGoalRegion = []

    @RRTComponent.catch_key_interrupt
    def start(self):
        for itera in range(self.maxIteration):
            self.cBestNow = self.cbest_single_tree(self.XInGoalRegion, self.xApp, itera, self.print_debug)

            xRand = self.uni_sampling()
            xNearest = self.nearest_node(self.treeVertex, xRand)
            xNew = self.steer(xNearest, xRand, self.eta)

            if self.is_collision_and_in_goal_region(xNearest, xNew, self.xGoal, self.distGoalToApp):
                continue
            xNew.parent = xNearest
            xNew.cost = xNew.parent.cost + self.cost_line(xNew, xNew.parent)
            xNearest.child.append(xNew)
            self.treeVertex.append(xNew)

            # in goal region
            if self.is_config_in_region_of_config(xNew, self.xApp, radius=self.nearGoalRadius):
                self.XInGoalRegion.append(xNew)

            if self.termination_check(self.XInGoalRegion):
                break

    def get_path(self):
        return self.search_best_cost_singledirection_path(backFromNode=self.xApp, treeVertexList=self.XInGoalRegion, attachNode=self.xGoal)

    def update_perf(self, timePlanningStart, timePlanningEnd):
        self.perf_matrix_update(tree1=self.treeVertex, tree2=None, timePlanningStart=timePlanningStart, timePlanningEnd=timePlanningEnd)

    def plot_tree(self, path, ax):
        self.plot_2d_obstacle(ax)
        self.plot_2d_single_tree(self.treeVertex, ax)
        self.plot_2d_path(path, ax)
        self.plot_2d_state_configuration(self.xStart, self.xApp, self.xGoal, ax)


class RRTBaseMulti(RRTComponent):

    def __init__(self, xStart, xAppList, xGoalList, eta, subEta, maxIteration, numDoF, envChoice, nearGoalRadius, rewireRadius, endIterationID, print_debug):
        super().__init__(eta, subEta, maxIteration, numDoF, envChoice, nearGoalRadius, rewireRadius, endIterationID, print_debug)
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
            self.cBestNow, self.xGoalBestIndex = self.cbest_single_tree_multi(self.XInGoalRegion, self.xAppList, itera, self.print_debug)

            biasIndex = np.random.randint(low=0, high=self.numGoal)
            xRand = self.bias_uniform_sampling(self.xAppList[biasIndex], len(self.XInGoalRegion[biasIndex]))
            xNearest = self.nearest_node(self.treeVertex, xRand)
            xNew = self.steer(xNearest, xRand, self.eta)
            if self.is_collision(xNearest, xNew):
                continue
            xNew.parent = xNearest
            xNew.cost = xNew.parent.cost + self.cost_line(xNew, xNew.parent)
            xNearest.child.append(xNew)
            self.treeVertex.append(xNew)

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

    def plot_tree(self, path, ax):
        self.plot_2d_obstacle(ax)
        self.plot_2d_single_tree(self.treeVertex, ax)
        self.plot_2d_path(path, ax)
        self.plot_2d_state_configuration(self.xStart, self.xAppList, self.xGoalList, ax)