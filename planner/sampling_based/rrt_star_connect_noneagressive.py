import os
import sys

sys.path.append(str(os.path.abspath(os.getcwd())))

from planner.sampling_based.rrt_component import Node, RRTComponent


class RRTStarConnectNoneAgressive(RRTComponent):

    def __init__(self, xStart, xApp, xGoal, config):
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

        # local sampling properties
        self.localOptEnable = config["localOptEnable"]
        if self.localOptEnable:
            self.anchorPath = None
            self.localPath = None
            self.numSegSamplingNode = None

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

            self.cBestNow = self.cbest_dual_tree(self.connectNodePair, itera) # save cost graph

            if self.localOptEnable:
                if len(self.connectNodePair) == 0:
                    xRand = self.uni_sampling()
                else:
                    xRand = self.local_path_sampling(self.anchorPath, self.localPath, self.numSegSamplingNode)
            else:
                xRand = self.uni_sampling()

            xNearest, vertexDistList = self.nearest_node(Ta, xRand, returnDistList=True)
            xNew, xNewIsxRand = self.steer(xNearest, xRand, self.eta, returnIsReached=True)

            if not self.is_collision_and_in_goal_region(xNearest, xNew, self.xGoal, self.distGoalToApp):
                xNew.parent = xNearest
                xNew.cost = xNew.parent.cost + self.cost_line(xNew, xNew.parent)
                xNearest.child.append(xNew)
                self.star_optimizer(Ta, xNew, self.rewireRadius, xNewIsxRand, vertexDistList)

                xNearestPrime = self.nearest_node(Tb, xNew)
                xNewPrime, xNewPrimeIsxNew = self.steer(xNearestPrime, xNew, self.eta, returnIsReached=True)

                if not self.is_collision_and_in_goal_region(xNearestPrime, xNewPrime, self.xGoal, self.distGoalToApp):
                    xNewPrime.parent = xNearestPrime
                    xNewPrime.cost = xNewPrime.parent.cost + self.cost_line(xNewPrime, xNewPrime.parent)
                    xNearestPrime.child.append(xNewPrime)
                    self.star_optimizer(Tb, xNewPrime, self.rewireRadius, xNewIsxRand=xNewPrimeIsxNew)

                    if self.distance_between_config(xNewPrime, xNew) < 1e-3:
                        if self.treeSwapFlag is True:
                            self.connectNodePair.append((xNew, xNewPrime))
                        elif self.treeSwapFlag is False:
                            self.connectNodePair.append((xNewPrime, xNew))
                        if self.localOptEnable:
                            if self.anchorPath is None:
                                self.localPath = self.search_best_cost_bidirection_path(self.connectNodePair)
                                self.numSegSamplingNode = len(self.localPath) - 1
                                self.anchorPath = self.segment_interpolation_between_config(self.localPath[0], self.localPath[-1], self.numSegSamplingNode, includexStart=True)

            if self.termination_check(self.connectNodePair):
                break

            self.tree_swap_flag()

    def get_path(self):
        return self.search_best_cost_bidirection_path(connectNodePairList=self.connectNodePair, attachNode=self.xGoal)


class RRTStarConnectNoneAgressiveMulti(RRTComponent):

    def __init__(self, xStart, xAppList, xGoalList, config):
        super().__init__(config)
        # start, aux, goal node
        self.xStart = Node(xStart)
        self.xGoalList = [Node(xGoali) for xGoali in xGoalList]
        self.xAppList = [Node(xAppi) for xAppi in xAppList]
        self.numGoal = len(self.xAppList)

        # planner properties
        self.treeVertexStart = [self.xStart]
        self.treeVertexGoal = [xAppi for xAppi in self.xAppList]
        self.distGoalToAppList = [self.distance_between_config(xAppi, xGoali) for xAppi, xGoali in zip(self.xAppList, self.xGoalList)]
        self.treeSwapFlag = True

        # local sampling properties
        self.localOptEnable = config["localOptEnable"]
        if self.localOptEnable:
            self.anchorPath = None
            self.localPath = None
            self.numSegSamplingNode = None

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

            self.cBestNow = self.cbest_dual_tree(self.connectNodePair, itera) # save cost graph

            if self.localOptEnable:
                if len(self.connectNodePair) == 0:
                    xRand = self.uni_sampling()
                else:
                    xRand = self.local_path_sampling(self.anchorPath, self.localPath, self.numSegSamplingNode)
            else:
                xRand = self.uni_sampling()

            xNearest, vertexDistList = self.nearest_node(Ta, xRand, returnDistList=True)
            xNew, xNewIsxRand = self.steer(xNearest, xRand, self.eta, returnIsReached=True)

            if not self.is_collision(xNearest, xNew):
                xNew.parent = xNearest
                xNew.cost = xNew.parent.cost + self.cost_line(xNew, xNew.parent)
                xNearest.child.append(xNew)
                self.star_optimizer(Ta, xNew, self.rewireRadius, xNewIsxRand, vertexDistList)

                xNearestPrime = self.nearest_node(Tb, xNew)
                xNewPrime, xNewPrimeIsxNew = self.steer(xNearestPrime, xNew, self.eta, returnIsReached=True)

                if not self.is_collision(xNearestPrime, xNewPrime):
                    xNewPrime.parent = xNearestPrime
                    xNewPrime.cost = xNewPrime.parent.cost + self.cost_line(xNewPrime, xNewPrime.parent)
                    xNearestPrime.child.append(xNewPrime)
                    self.star_optimizer(Tb, xNewPrime, self.rewireRadius, xNewIsxRand=xNewPrimeIsxNew)

                    if self.distance_between_config(xNewPrime, xNew) < 1e-3:
                        if self.treeSwapFlag is True:
                            self.connectNodePair.append((xNew, xNewPrime))
                        elif self.treeSwapFlag is False:
                            self.connectNodePair.append((xNewPrime, xNew))
                        if self.localOptEnable:
                            if self.anchorPath is None:
                                self.localPath = self.search_best_cost_bidirection_path(self.connectNodePair)
                                self.numSegSamplingNode = len(self.localPath) - 1
                                self.anchorPath = self.segment_interpolation_between_config(self.localPath[0], self.localPath[-1], self.numSegSamplingNode, includexStart=True)

            if self.termination_check(self.connectNodePair):
                break

            self.tree_swap_flag()

    def get_path(self):
        path = self.search_best_cost_bidirection_path(connectNodePairList=self.connectNodePair)
        xGoalIndex = self.xAppList.index(path[-1])
        path = path + [self.xGoalList[xGoalIndex]]
        return path