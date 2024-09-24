""" 
Devplanner with RRT-Connect assisted Informed-RRT* with reject goal sampling and multiGoal

"""
import os
import sys
sys.path.append(str(os.path.abspath(os.getcwd())))

import numpy as np
from planner.rrt_component import Node, RRTComponent


class RRTMyDevMultiGoal(RRTComponent):
    def __init__(self, xStart, xApp, xGoal, eta, subEta, maxIteration, numDoF, envChoice, nearGoalRadius) -> None:
        super().__init__(numDoF=numDoF, envChoice=envChoice)
        # start, aux, goal node
        self.xStart = Node(xStart)
        self.xGoalList = [Node(xGoali) for xGoali in xGoal]
        self.xAppList = [Node(xAppi) for xAppi in xApp]

        self.eta = eta
        self.subEta = subEta
        self.nearGoalRadius = nearGoalRadius
        self.maxIteration = maxIteration
        self.treeVertexStart = [self.xStart]
        self.treeVertexGoalList = [[xAppi] for xAppi in self.xAppList]
        self.treeSwapFlag = True
        self.connectNodeStart = [None]*len(self.xAppList)
        self.connectNodeGoal = [None]*len(self.xAppList)
        self.rewireRadius = None
        self.initialCost = [None]*len(self.xAppList)
        self.distGoalToAppList = [self.distance_between_config(xAppi, xGoali) for xAppi, xGoali in zip(self.xAppList, self.xGoalList)]
        self.targetIndexNow = 0

        # informed sampling properties
        self.CList = [self.rotation_to_world(self.xStart, xApp) for xApp in self.xAppList]
        self.cMinList = [self.distance_between_config(self.xStart, xApp) for xApp in self.xAppList]
        self.xCenterList = [(self.xStart.config + xApp.config) / 2 for xApp in self.xAppList]
        self.XSolnList = [[] for _ in range(len(self.xAppList))]

    @RRTComponent.catch_key_interrupt
    def start(self):
        for itera in range(self.maxIteration):
            print(itera)
            if self.treeSwapFlag is True:
                xRand = self.uni_sampling()
                xNearest = self.nearest_node(self.treeVertexStart, xRand)
                xNew = self.steer(xNearest, xRand, self.eta)

                if not self.is_collision_and_in_goal_region(xNearest, xNew, self.xGoalList[self.targetIndexNow], self.distGoalToAppList[self.targetIndexNow]):
                    xNew.parent = xNearest
                    xNew.cost = xNew.parent.cost + self.cost_line(xNew, xNew.parent)
                    xNearest.child.append(xNew)
                    self.treeVertexStart.append(xNew)
                    xNearestPrime = self.nearest_node(self.treeVertexGoalList[self.targetIndexNow], xNew)
                    xNewPrime = self.steer(xNearestPrime, xNew, self.eta)

                    if not self.is_collision_and_in_goal_region(xNearestPrime, xNewPrime, self.xGoalList[self.targetIndexNow], self.distGoalToAppList[self.targetIndexNow]):
                        xNewPrime.parent = xNearestPrime
                        xNewPrime.cost = xNewPrime.parent.cost + self.cost_line(xNewPrime, xNewPrime.parent)
                        xNearestPrime.child.append(xNewPrime)
                        self.treeVertexGoalList[self.targetIndexNow].append(xNewPrime)

                        while True:
                            xNewPPrime = self.steer(xNewPrime, xNew, self.eta)

                            if self.is_collision_and_in_goal_region(xNewPrime, xNewPPrime, self.xGoalList[self.targetIndexNow], self.distGoalToAppList[self.targetIndexNow]):
                                break

                            if self.distance_between_config(xNewPPrime, xNew) < 1e-3:
                                self.connectNodeGoal[self.targetIndexNow] = xNewPrime
                                self.connectNodeStart[self.targetIndexNow] = xNew
                                self.initialCost[self.targetIndexNow] = xNew.cost + xNewPrime.cost + self.cost_line(xNew, xNewPrime)
                                self.targetIndexNow += 1
                                break

                            else:
                                xNewPPrime.parent = xNewPrime
                                xNewPPrime.cost = xNewPPrime.parent.cost + self.cost_line(xNewPPrime, xNewPPrime.parent)
                                xNewPrime.child.append(xNewPPrime)
                                self.treeVertexGoalList[self.targetIndexNow].append(xNewPPrime)
                                xNewPrime = xNewPPrime

                if self.targetIndexNow == len(self.xAppList):
                    break

                self.tree_swap_flag()

            elif self.treeSwapFlag is False:
                xRand = self.uni_sampling()
                xNearest = self.nearest_node(self.treeVertexGoalList[self.targetIndexNow], xRand)
                xNew = self.steer(xNearest, xRand, self.eta)

                if not self.is_collision_and_in_goal_region(xNearest, xNew, self.xGoalList[self.targetIndexNow], self.distGoalToAppList[self.targetIndexNow]):
                    xNew.parent = xNearest
                    xNew.cost = xNew.parent.cost + self.cost_line(xNew, xNew.parent)
                    xNearest.child.append(xNew)
                    self.treeVertexGoalList[self.targetIndexNow].append(xNew)
                    xNearestPrime = self.nearest_node(self.treeVertexStart, xNew)
                    xNewPrime = self.steer(xNearestPrime, xNew, self.eta)

                    if not self.is_collision_and_in_goal_region(xNearestPrime, xNewPrime, self.xGoalList[self.targetIndexNow], self.distGoalToAppList[self.targetIndexNow]):
                        xNewPrime.parent = xNearestPrime
                        xNewPrime.cost = xNewPrime.parent.cost + self.cost_line(xNewPrime, xNewPrime.parent)
                        xNearestPrime.child.append(xNewPrime)
                        self.treeVertexStart.append(xNewPrime)

                        while True:
                            xNewPPrime = self.steer(xNewPrime, xNew, self.eta)

                            if self.is_collision_and_in_goal_region(xNewPrime, xNewPPrime, self.xGoalList[self.targetIndexNow], self.distGoalToAppList[self.targetIndexNow]):
                                break

                            if self.distance_between_config(xNewPPrime, xNew) < 1e-3:
                                self.connectNodeGoal[self.targetIndexNow] = xNew
                                self.connectNodeStart[self.targetIndexNow] = xNewPrime
                                self.initialCost[self.targetIndexNow] = xNew.cost + xNewPrime.cost + self.cost_line(xNew, xNewPrime)
                                self.targetIndexNow += 1
                                break

                            else:
                                xNewPPrime.parent = xNewPrime
                                xNewPPrime.cost = xNewPPrime.parent.cost + self.cost_line(xNewPPrime, xNewPPrime.parent)
                                xNewPrime.child.append(xNewPPrime)
                                self.treeVertexStart.append(xNewPPrime)
                                xNewPrime = xNewPPrime

                if self.targetIndexNow == len(self.xAppList):
                    break

                self.tree_swap_flag()

        sortedHighToLow = sorted(range(len(self.initialCost)), key=lambda i: self.initialCost[i], reverse=True)
        iterationPerPath = int((self.maxIteration - itera)/len(self.xAppList))

        for ind in sortedHighToLow:
            xApp = self.xAppList[ind]
            xGoal = self.xGoalList[ind]
            cost = self.initialCost[ind]
            for remainItera in range(iterationPerPath):
                print(remainItera)
                if len(self.XSolnList[ind]) == 0:
                    cBest = cost
                else:
                    xSolnCost = [xSoln.cost + self.cost_line(xSoln, xApp) for xSoln in self.XSolnList[ind]]
                    # print(f"==>> xSolnCost: \n{xSolnCost}")
                    cBest = min(xSolnCost)
                    # if cBest < self.cBestPrevious : # this have nothing to do with planning itself, just for record performance data only
                    #     self.perfMatrix["Cost Graph"].append((itera, cBest))
                    #     self.cBestPrevious = cBest

                xRand = self.informed_sampling(self.xCenterList[ind], cBest, self.cMinList[ind], self.CList[ind])
                xNearest = self.nearest_node(self.treeVertexStart, xRand)
                xNew = self.steer(xNearest, xRand, self.eta)
                if self.is_collision_and_in_goal_region(xNearest, xNew, xGoal, self.distGoalToAppList[ind]):
                    continue
                xNew.parent = xNearest
                xNew.cost = xNew.parent.cost + self.cost_line(xNew, xNew.parent)
                xNearest.child.append(xNew) #1

                XNear = self.near(self.treeVertexStart, xNew, self.rewireRadius)
                cMin = xNew.cost
                XNearCostToxNew = [xNear.cost + self.cost_line(xNear, xNew) for xNear in XNear]
                XNearCollisionState = [None]*len(XNear)
                XNearCostToxNewSortedIndex = np.argsort(XNearCostToxNew)
                for index in XNearCostToxNewSortedIndex:
                    if XNearCostToxNew[index] < cMin:
                        collisionState = self.is_connect_config_in_collision(XNear[index], xNew)
                        XNearCollisionState[index] = collisionState
                        if not collisionState:
                            xNearest.child.remove(xNew) # 2
                            xNew.parent = XNear[index]
                            xNew.cost = XNearCostToxNew[index]
                            XNear[index].child.append(xNew)  # 3
                            break

                self.treeVertexStart.append(xNew)

                for index, xNear in enumerate(XNear):
                    cNew = xNew.cost + self.cost_line(xNew, xNear)
                    if cNew < xNear.cost:
                        if XNearCollisionState[index] is None:
                            collisionState = self.is_connect_config_in_collision(xNear, xNew)
                        else:
                            collisionState = XNearCollisionState[index]
                        if collisionState is False:
                            xNear.parent.child.remove(xNear) # 4
                            xNear.parent = xNew
                            xNear.cost = cNew
                            xNew.child.append(xNear) # 5
                            self.update_child_cost(xNear) #6

                # in approach region
                if self.is_config_in_region_of_config(xNew, xApp, radius=self.nearGoalRadius):
                    self.XSolnList[ind].append(xNew)
    
    def plot_tree(self, path1, path2, path3, ax):
        self.plot_2d_obstacle(ax)
        self.plot_2d_single_tree(self.treeVertexStart, ax)
        self.plot_2d_state_configuration(self.xStart, self.xAppList, self.xGoalList, ax)
        self.plot_2d_path(path1, ax)
        self.plot_2d_path(path2, ax)
        self.plot_2d_path(path3, ax)
