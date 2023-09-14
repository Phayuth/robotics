import os
import sys

wd = os.path.abspath(os.getcwd())
sys.path.append(str(wd))

import numpy as np
from planner_dev.rrt_component import Node, RRTComponent


class RRTInformedConnectDev(RRTComponent):

    def __init__(self, xStart, xApp, xGoal, eta, subEta, maxIteration, numDoF, envChoice) -> None:
        super().__init__(NumDoF=numDoF, EnvChoice=envChoice)
        # start, aux, goal node
        self.xStart = Node(xStart)
        self.xGoal = Node(xGoal)
        self.xApp = Node(xApp)

        self.eta = eta
        self.subEta = subEta
        self.maxIteration = maxIteration
        self.treeVertexStart = [self.xStart]
        self.treeVertexGoal = [self.xApp]
        self.treeSwapFlag = True
        self.connectNodePair = []  # (connectStart, connectGoal)
        self.distGoalToApp = self.distance_between_config(self.xGoal, self.xApp)
        self.rewireRadius = None

        # informed sampling properties
        self.C = self.rotation_to_world(self.xStart, self.xGoal) # hyperellipsoid rotation axis
        self.cMin = self.distance_between_config(self.xStart, self.xGoal)
        self.xCenter = (self.xStart.config + self.xApp.config) / 2

    @RRTComponent.catch_key_interrupt
    def start(self):
        for itera in range(self.maxIteration):
            print(itera)
            if self.treeSwapFlag is True:
                Ta = self.treeVertexStart
                Tb = self.treeVertexGoal
            elif self.treeSwapFlag is False:
                Ta = self.treeVertexGoal
                Tb = self.treeVertexStart

            cBest = self.dual_tree_cbest(self.connectNodePair, itera)

            if cBest == np.inf:
                xRand = self.uni_sampling()
            elif cBest < np.inf:
                xRand = self.informed_sampling(self.xCenter, cBest, self.cMin, self.C)

            xNearest, vertexDistList = self.nearest_node(Ta, xRand, returnDistList=True)
            xNew, xNewIsxRand = self.steer(xNearest, xRand, self.eta, returnxNewIsxRand=True)

            if not self.is_collision_and_in_goal_region(xNearest, xNew, self.xGoal, self.distGoalToApp):
                xNew.parent = xNearest
                xNew.cost = xNew.parent.cost + self.cost_line(xNew, xNew.parent)
                xNearest.child.append(xNew)
                self.star_optimizer(Ta, xNew, self.rewireRadius, xNewIsxRand, vertexDistList)

                xNearestPrime = self.nearest_node(Tb, xNew)
                xNewPrime, xNewPrimeIsxNew = self.steer(xNearestPrime, xNew, self.eta, returnxNewIsxRand=True)

                if not self.is_collision_and_in_goal_region(xNearestPrime, xNewPrime, self.xGoal, self.distGoalToApp):
                    xNewPrime.parent = xNearestPrime
                    xNewPrime.cost = xNewPrime.parent.cost + self.cost_line(xNewPrime, xNewPrime.parent)
                    xNearestPrime.child.append(xNewPrime)
                    self.star_optimizer(Tb, xNewPrime, self.rewireRadius, xNewIsxRand=xNewPrimeIsxNew)

                    while True:
                        xNewPPrime, xNewPPrimeIsxNewPrime = self.steer(xNewPrime, xNew, self.eta, returnxNewIsxRand=True)

                        if self.is_collision_and_in_goal_region(xNewPrime, xNewPPrime, self.xGoal, self.distGoalToApp):
                            break

                        if self.distance_between_config(xNewPPrime, xNew) < 1e-3:
                            if self.treeSwapFlag is True:
                                self.connectNodePair.append((xNew, xNewPrime))
                            elif self.treeSwapFlag is False:
                                self.connectNodePair.append((xNewPrime, xNew))
                            break

                        else:
                            xNewPPrime.parent = xNewPrime
                            xNewPPrime.cost = xNewPPrime.parent.cost + self.cost_line(xNewPPrime, xNewPPrime.parent)
                            xNewPrime.child.append(xNewPPrime)
                            self.star_optimizer(Tb, xNewPPrime, self.rewireRadius, xNewPPrimeIsxNewPrime)
                            xNewPrime = xNewPPrime

            self.tree_swap_flag()

    def plot_tree(self, path, ax):
        self.plot_2d_obstacle(ax)
        self.plot_2d_dual_tree(self.treeVertexStart, self.treeVertexGoal, ax)
        self.plot_2d_path(path, ax)
        self.plot_2d_state_configuration(self.xStart, self.xApp, self.xGoal, ax)
