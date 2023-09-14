import os
import sys

wd = os.path.abspath(os.getcwd())
sys.path.append(str(wd))

from planner_dev.rrt_component import Node, RRTComponent


class RRTConnectDev(RRTComponent):

    def __init__(self, xStart, xApp, xGoal, eta, subEta, maxIteration, numDoF, envChoice):
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
        self.connectNodeStart = None
        self.connectNodeGoal = None
        self.distGoalToApp = self.distance_between_config(self.xGoal, self.xApp)

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

            xRand = self.uni_sampling()
            xNearest = self.nearest_node(Ta, xRand)
            xNew = self.steer(xNearest, xRand, self.eta)

            if not self.is_collision_and_in_goal_region(xNearest, xNew, self.xGoal, self.distGoalToApp):
                xNew.parent = xNearest
                xNew.cost = xNew.parent.cost + self.cost_line(xNew, xNew.parent)
                xNearest.child.append(xNew)
                Ta.append(xNew)
                xNearestPrime = self.nearest_node(Tb, xNew)
                xNewPrime = self.steer(xNearestPrime, xNew, self.eta)

                if not self.is_collision_and_in_goal_region(xNearestPrime, xNewPrime, self.xGoal, self.distGoalToApp):
                    xNewPrime.parent = xNearestPrime
                    xNewPrime.cost = xNewPrime.parent.cost + self.cost_line(xNewPrime, xNewPrime.parent)
                    xNearestPrime.child.append(xNewPrime)
                    Tb.append(xNewPrime)

                    while True:
                        xNewPPrime = self.steer(xNewPrime, xNew, self.eta)

                        if self.is_collision_and_in_goal_region(xNewPrime, xNewPPrime, self.xGoal, self.distGoalToApp):
                            break

                        if self.distance_between_config(xNewPPrime, xNew) < 1e-3:
                            if self.treeSwapFlag is True:
                                self.connectNodeGoal = xNewPrime
                                self.connectNodeStart = xNew
                            elif self.treeSwapFlag is False:
                                self.connectNodeGoal = xNew
                                self.connectNodeStart = xNewPrime
                            break

                        else:
                            xNewPPrime.parent = xNewPrime
                            xNewPPrime.cost = xNewPPrime.parent.cost + self.cost_line(xNewPPrime, xNewPPrime.parent)
                            xNewPrime.child.append(xNewPPrime)
                            Tb.append(xNewPPrime)
                            xNewPrime = xNewPPrime

            if self.connectNodeGoal is not None and self.connectNodeStart is not None:
                cBest = self.connectNodeStart.cost + self.connectNodeGoal.cost + self.cost_line(self.connectNodeStart, self.connectNodeGoal)
                self.perfMatrix["Cost Graph"].append((itera, cBest))
                self.reparent_merge_tree(xTobeParent=self.connectNodeStart, xNow=self.connectNodeGoal, treeToAddTo=self.treeVertexStart)
                break

            self.tree_swap_flag()

    def plot_tree(self, path, ax):
        self.plot_2d_obstacle(ax)
        self.plot_2d_dual_tree(self.treeVertexStart, self.treeVertexGoal, ax)
        self.plot_2d_state_configuration(self.xStart, self.xApp, self.xGoal, ax)
        self.plot_2d_path(path, ax)