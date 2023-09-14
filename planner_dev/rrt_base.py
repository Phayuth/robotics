import os
import sys

wd = os.path.abspath(os.getcwd())
sys.path.append(str(wd))

import numpy as np
from planner_dev.rrt_component import Node, RRTComponent


class RRTBase(RRTComponent):

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
        self.XInGoalRegion = []

    @RRTComponent.catch_key_interrupt
    def start(self):
        for itera in range(self.maxIteration):
            print(itera)
            _ = self.single_tree_cbest(self.XInGoalRegion, self.xApp, itera)

            xRand = self.uni_sampling()
            xNearest = self.nearest_node(self.treeVertex, xRand)
            xNew = self.steer(xNearest, xRand, self.eta)

            if self.is_collision_and_in_goal_region(xNearest, xNew, self.xGoal, self.distGoalToApp):
                continue
            xNew.parent = xNearest
            xNew.cost = xNew.parent.cost + self.cost_line(xNew, xNew.parent)
            xNearest.child.append(xNew)
            self.treeVertex.append(xNew)

            # in goal region, this have nothing to do with planning itself, just for record performance data only
            if self.is_config_in_region_of_config(xNew, self.xApp, radius=self.nearGoalRadius):
                self.XInGoalRegion.append(xNew)

    def plot_tree(self, path, ax):
        self.plot_2d_obstacle(ax)
        self.plot_2d_single_tree(self.treeVertex, ax)
        self.plot_2d_path(path, ax)
        self.plot_2d_state_configuration(self.xStart, self.xApp, self.xGoal, ax)
