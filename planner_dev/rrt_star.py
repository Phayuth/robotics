import os
import sys
wd = os.path.abspath(os.getcwd())
sys.path.append(str(wd))

import numpy as np
from planner_dev.rrt_component import Node, RRTComponent


class RRTStarDev(RRTComponent):

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

    # def rrt_star(self):
    #     for itera in range(self.maxIteration):
    #         print(itera)
    #         # this have nothing to do with planning itself, just for record performance data only
    #         if len(self.XInGoalRegion) == 0:
    #             cBest = np.inf
    #             cBestPrevious = np.inf
    #         else:
    #             xSolnCost = [xSoln.parent.cost + self.cost_line(xSoln.parent, xSoln) + self.cost_line(xSoln, self.xApp) for xSoln in self.XInGoalRegion]
    #             print(f"==>> xSolnCost: \n{xSolnCost}")
    #             cBest = min(xSolnCost)
    #             if cBest < cBestPrevious:
    #                 self.perfMatrix["Cost Graph"].append((itera, cBest))
    #                 cBestPrevious = cBest

    #         xRand = self.uni_sampling()
    #         xNearest = self.nearest_node(self.treeVertex, xRand)
    #         xNew = self.steer(xNearest, xRand, self.eta)
    #         if self.is_collision_and_in_goal_region(xNearest, xNew, self.xGoal, self.distGoalToApp):
    #             continue
    #         xNew.parent = xNearest
    #         xNew.cost = xNew.parent.cost + self.cost_line(xNew, xNew.parent)
    #         XNear = self.near(self.treeVertex, xNew, self.rewireRadius)
    #         xMin = xNew.parent
    #         cMin = xMin.cost + self.cost_line(xMin, xNew)
    #         for xNear in XNear:
    #             if self.is_connect_config_in_collision(xNear, xNew):
    #                 continue

    #             cNew = xNear.cost + self.cost_line(xNear, xNew)
    #             if cNew < cMin:
    #                 xMin = xNear
    #                 cMin = cNew

    #         xNew.parent = xMin
    #         xNew.cost = cMin
    #         self.treeVertex.append(xNew)

    #         for xNear in XNear:
    #             if self.is_connect_config_in_collision(xNear, xNew):
    #                 continue
    #             cNear = xNear.cost
    #             cNew = xNew.cost + self.cost_line(xNew, xNear)
    #             if cNew < cNear:
    #                 xNear.parent = xNew
    #                 xNear.cost = xNew.cost + self.cost_line(xNew, xNear)

    #         # in goal region, this have nothing to do with planning itself, just for record performance data only
    #         if self.is_config_in_region_of_config(xNew, self.xApp, radius=self.nearGoalRadius):
    #             self.XInGoalRegion.append(xNew)

    #     return itera

    @RRTComponent.catch_key_interrupt
    def start(self):
        for itera in range(self.maxIteration):
            print(itera)
            _ = self.single_tree_cbest(self.XInGoalRegion, self.xApp, itera) # save cost graph

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

            # in goal region, this have nothing to do with planning itself, just for record performance data only
            if self.is_config_in_region_of_config(xNew, self.xApp, radius=self.nearGoalRadius):
                self.XInGoalRegion.append(xNew)
    
    def plot_tree(self, path, ax):
        self.plot_2d_obstacle(ax)
        self.plot_2d_single_tree(self.treeVertex, ax)
        self.plot_2d_path(path, ax)
        self.plot_2d_state_configuration(self.xStart, self.xApp, self.xGoal, ax)
