import os
import sys
wd = os.path.abspath(os.getcwd())
sys.path.append(str(wd))

import numpy as np
from planner_dev.rrt_component import Node, RRTComponent


class RRTStarLocalOpt(RRTComponent):

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

        self.anchorPath = None
        self.localPath = None
        self.numSegSamplingNode = None

    @RRTComponent.catch_key_interrupt
    def start(self):
        for itera in range(self.maxIteration):
            print(itera)
            _ = self.single_tree_cbest(self.XInGoalRegion, self.xApp, itera) # save cost graph

            if len(self.XInGoalRegion) == 0:
                xRand = self.bias_uniform_sampling(self.xApp, len(self.XInGoalRegion))
            else:
                xRand = self.local_path_sampling(self.anchorPath, self.localPath, self.numSegSamplingNode)
            
            xNearest, vertexDistList = self.nearest_node(self.treeVertex, xRand, returnDistList=True)
            xNew, xNewIsxRand = self.steer(xNearest, xRand, self.eta, returnxNewIsxRand=True)
            if self.is_collision_and_in_goal_region(xNearest, xNew, self.xGoal, self.distGoalToApp):
                continue
            xNew.parent = xNearest
            xNew.cost = xNew.parent.cost + self.cost_line(xNew, xNew.parent)
            xNearest.child.append(xNew)

            # rrtstar
            self.star_optimizer(self.treeVertex, xNew, self.rewireRadius, xNewIsxRand, vertexDistList)

            # in goal region, this have nothing to do with planning itself, just for record performance data only
            if self.is_config_in_region_of_config(xNew, self.xApp, radius=self.nearGoalRadius):
                self.XInGoalRegion.append(xNew)
                if self.anchorPath is None:
                    self.localPath = self.search_backtrack_single_directional_path(self.XInGoalRegion[0], self.xApp)
                    self.numSegSamplingNode = len(self.localPath)-1
                    self.anchorPath = self.segment_interpolation_between_config(self.xStart, self.xApp, self.numSegSamplingNode, includexStart=True)
    
    def plot_tree(self, path, ax):
        self.plot_2d_obstacle(ax)
        self.plot_2d_single_tree(self.treeVertex, ax)
        self.plot_2d_path(path, ax)
        self.plot_2d_state_configuration(self.xStart, self.xApp, self.xGoal, ax)
