import os
import sys

sys.path.append(str(os.path.abspath(os.getcwd())))

import numpy as np
from planner.sampling_based.rrt_component import Node, RRTComponent


class RRTInformedConnectQuickNoneAgressive(RRTComponent):

    def __init__(self, xStart, xApp, xGoal, config) -> None:
        super().__init__(config)
        # start, aux, goal node
        self.xStart = Node(xStart)
        self.xGoal = Node(xGoal)
        self.xApp = Node(xApp)

        # planner properties
        self.treeVertexStart = [self.xStart]
        self.treeVertexGoal = [self.xApp]
        self.distGoalToApp = self.distance_between_config(self.xGoal, self.xApp)
        self.treeSwapFlag = True

        # informed sampling properties
        self.C = self.rotation_to_world(self.xStart, self.xGoal) # hyperellipsoid rotation axis
        self.cMin = self.distance_between_config(self.xStart, self.xGoal)
        self.xCenter = (self.xStart.config + self.xApp.config) / 2

        # solutions
        self.connectNodePair = []  # (connectStart, connectGoal)

    @RRTComponent.catch_key_interrupt
    def start(self):
        for itera in range(self.maxIteration):
            if self.treeSwapFlag is True:
                Ta = self.treeVertexStart
                Tb = self.treeVertexGoal
            elif self.treeSwapFlag is False:
                Ta = self.treeVertexGoal
                Tb = self.treeVertexStart

            self.cBestNow = self.cbest_dual_tree(self.connectNodePair, itera)

            if self.cBestNow == np.inf:
                xRand = self.uni_sampling()
            elif self.cBestNow < np.inf:
                xRand = self.informed_sampling(self.xCenter, self.cBestNow, self.cMin, self.C)

            xNearest, vertexDistList = self.nearest_node(Ta, xRand, returnDistList=True)
            xNew, xNewIsxRand = self.steer(xNearest, xRand, self.eta, returnIsReached=True)

            if not self.is_collision_and_in_goal_region(xNearest, xNew, self.xGoal, self.distGoalToApp):
                xNew.parent = xNearest
                xNew.cost = xNew.parent.cost + self.cost_line(xNew, xNew.parent)
                xNearest.child.append(xNew)
                self.star_quick_optimizer(Ta, xNew, self.rewireRadius, xNewIsxRand, vertexDistList)

                xNearestPrime = self.nearest_node(Tb, xNew)
                xNewPrime, xNewPrimeIsxNew = self.steer(xNearestPrime, xNew, self.eta, returnIsReached=True)

                if not self.is_collision_and_in_goal_region(xNearestPrime, xNewPrime, self.xGoal, self.distGoalToApp):
                    xNewPrime.parent = xNearestPrime
                    xNewPrime.cost = xNewPrime.parent.cost + self.cost_line(xNewPrime, xNewPrime.parent)
                    xNearestPrime.child.append(xNewPrime)
                    self.star_quick_optimizer(Tb, xNewPrime, self.rewireRadius, xNewPrimeIsxNew)

                    if self.distance_between_config(xNewPrime, xNew) < 1e-3:
                        if self.treeSwapFlag is True:
                            self.connectNodePair.append((xNew, xNewPrime))
                        elif self.treeSwapFlag is False:
                            self.connectNodePair.append((xNewPrime, xNew))
                        break

            if self.termination_check(self.connectNodePair):
                break

            self.tree_swap_flag()

    def get_path(self):
        return self.search_best_cost_bidirection_path(connectNodePairList=self.connectNodePair, attachNode=self.xGoal)