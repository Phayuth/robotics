""" 
Devplanner with RRT-Connect assisted Informed-RRT* with reject goal sampling and multiGoal

"""
import os
import sys
wd = os.path.abspath(os.getcwd())
sys.path.append(str(wd))

import time
import numpy as np
from planner_dev.rrt_component import Node, RRTComponent


class RRTInformedStarDev2D(RRTComponent):
    def __init__(self, xStart, xApp=[], xGoal=[], eta=0.3, maxIteration=1000) -> None:
        super().__init__(NumDoF=2, EnvChoice="Planar")
        self.xStart = Node(xStart)

        self.xGoal1 = Node(xGoal[0])
        self.xGoal2 = Node(xGoal[1])
        self.xGoal3 = Node(xGoal[2])

        self.xApp1 = Node(xApp[0])
        self.xApp2 = Node(xApp[1])
        self.xApp3 = Node(xApp[2])

        self.eta = eta
        self.subEta = 0.05
        self.maxIteration = maxIteration

        self.treeVertexStart = [self.xStart]
        self.treeVertexGoal = [[self.xApp1], [self.xApp2], [self.xApp3]]

        self.xAppList = [self.xApp1, self.xApp2, self.xApp3]
        self.xGoalList = [self.xGoal1, self.xGoal2, self.xGoal3]
        self.distGoalToAppList = [self.distance_between_config(self.xGoalList[i], self.xAppList[i]) for i in range(len(self.xAppList))]
        self.connectNodeStart = [None, None, None]
        self.connectNodeGoal = [None, None, None]
        self.targetIndexNow = 0

        self.treeSwapFlag = True
        self.rewireRadius = 0.5

        self.XSoln = []

    def planning(self):
        timePlanningStart = time.perf_counter_ns()
        itera = self.planner_rrt_connect_app()
        path1 = self.search_backtrack_bidirectional_path(self.connectNodeStart[0], self.connectNodeGoal[0], self.xGoal1)
        path2 = self.search_backtrack_bidirectional_path(self.connectNodeStart[1], self.connectNodeGoal[1], self.xGoal2)
        path3 = self.search_backtrack_bidirectional_path(self.connectNodeStart[2], self.connectNodeGoal[2], self.xGoal3)
        timePlanningEnd = time.perf_counter_ns()

        # record performance
        self.perfMatrix["Planner Name"] = "Proposed + MultiGoal"
        self.perfMatrix["Parameters"]["eta"] = self.eta 
        self.perfMatrix["Parameters"]["Max Iteration"] = self.maxIteration
        self.perfMatrix["Parameters"]["Rewire Radius"] = self.rewireRadius
        self.perfMatrix["Number of Node"] = len(self.treeVertexStart)
        self.perfMatrix["Total Planning Time"] = (timePlanningEnd-timePlanningStart) * 1e-9
        self.perfMatrix["Planning Time Only"] = self.perfMatrix["Total Planning Time"] - self.perfMatrix["KCD Time Spend"]* 1e-9
        self.perfMatrix["Average KCD Time"] = self.perfMatrix["KCD Time Spend"] / self.perfMatrix["Number of Collision Check"]
        
        return path1, path2, path3

    def planner_rrt_connect_app(self):
        for itera in range(self.maxIteration):
            print(itera)
            if self.treeSwapFlag is True: # Init tree side
                xRand = self.uni_sampling()
                xNearest = self.nearest_node(self.treeVertexStart, xRand)
                xNew = self.steer(xNearest, xRand, self.eta)

                if not self.is_collision_and_in_goal_region(xNearest, xNew, self.xGoalList[self.targetIndexNow], self.distGoalToAppList[self.targetIndexNow]):
                    xNew.parent = xNearest
                    xNew.cost = xNew.parent.cost + self.cost_line(xNew, xNew.parent)
                    self.treeVertexStart.append(xNew)
                    xNearestPrime = self.nearest_node(self.treeVertexGoal[self.targetIndexNow], xNew)
                    xNewPrime = self.steer(xNearestPrime, xNew, self.eta)

                    if not self.is_collision_and_in_goal_region(xNearestPrime, xNewPrime, self.xGoalList[self.targetIndexNow], self.distGoalToAppList[self.targetIndexNow]):
                        xNewPrime.parent = xNearestPrime
                        xNewPrime.cost = xNewPrime.parent.cost + self.cost_line(xNewPrime, xNewPrime.parent)
                        self.treeVertexGoal[self.targetIndexNow].append(xNewPrime)

                        while True:
                            xNewPPrime = self.steer(xNewPrime, xNew, self.eta)

                            if self.is_collision_and_in_goal_region(xNewPrime, xNewPPrime, self.xGoalList[self.targetIndexNow], self.distGoalToAppList[self.targetIndexNow]):
                                break

                            # if the 2 node meet, then break
                            if self.distance_between_config(xNewPPrime, xNew) < 1e-3:
                                self.connectNodeGoal[self.targetIndexNow] = xNewPrime
                                self.connectNodeStart[self.targetIndexNow] = xNew
                                self.targetIndexNow += 1
                                break  # break out sampling and break out of planner

                            # if not collision then free to add
                            else:
                                xNewPPrime.parent = xNewPrime
                                xNewPPrime.cost = xNewPPrime.parent.cost + self.cost_line(xNewPPrime, xNewPPrime.parent)
                                self.treeVertexGoal[self.targetIndexNow].append(xNewPPrime)
                                xNewPrime = xNewPPrime

                # if self.connectNodeGoal is not None and self.connectNodeStart is not None:
                #     break
                if self.targetIndexNow == len(self.xAppList):
                    break

                self.tree_swap_flag()

            elif self.treeSwapFlag is False:  # App tree side
                xRand = self.uni_sampling()
                xNearest = self.nearest_node(self.treeVertexGoal[self.targetIndexNow], xRand)
                xNew = self.steer(xNearest, xRand, self.eta)
                
                if not self.is_collision_and_in_goal_region(xNearest, xNew, self.xGoalList[self.targetIndexNow], self.distGoalToAppList[self.targetIndexNow]):
                    xNew.parent = xNearest
                    xNew.cost = xNew.parent.cost + self.cost_line(xNew, xNew.parent)
                    self.treeVertexGoal[self.targetIndexNow].append(xNew)
                    xNearestPrime = self.nearest_node(self.treeVertexStart, xNew)
                    xNewPrime = self.steer(xNearestPrime, xNew, self.eta)
                    
                    if not self.is_collision_and_in_goal_region(xNearestPrime, xNewPrime, self.xGoalList[self.targetIndexNow], self.distGoalToAppList[self.targetIndexNow]):
                        xNewPrime.parent = xNearestPrime
                        xNewPrime.cost = xNewPrime.parent.cost + self.cost_line(xNewPrime, xNewPrime.parent)
                        self.treeVertexStart.append(xNewPrime)

                        while True:
                            xNewPPrime = self.steer(xNewPrime, xNew, self.eta)

                            # if there is collision then break and if the node and connection of node to parent is inside region
                            if self.is_collision_and_in_goal_region(xNewPrime, xNewPPrime, self.xGoalList[self.targetIndexNow], self.distGoalToAppList[self.targetIndexNow]):
                                break

                            # if the 2 node meet, then break
                            if self.distance_between_config(xNewPPrime, xNew) < 1e-3:
                                self.connectNodeGoal[self.targetIndexNow] = xNew
                                self.connectNodeStart[self.targetIndexNow] = xNewPrime
                                self.targetIndexNow += 1
                                break

                            # if not collision then free to add
                            else:
                                xNewPPrime.parent = xNewPrime
                                xNewPPrime.cost = xNewPPrime.parent.cost + self.cost_line(xNewPPrime, xNewPPrime.parent)
                                self.treeVertexStart.append(xNewPPrime)
                                xNewPrime = xNewPPrime

                # if self.connectNodeGoal is not None and self.connectNodeStart is not None:
                #     break
                if self.targetIndexNow == len(self.xAppList):
                    break

                self.tree_swap_flag()

        return itera

    def plot_tree(self, path1=None, path2=None, path3=None):
        for vertex in self.treeVertexStart:
            if vertex.parent == None:
                pass
            else:
                plt.plot([vertex.config[0], vertex.parent.config[0]], [vertex.config[1], vertex.parent.config[1]], color="darkgray")

        for vertex in self.treeVertexGoal[0]:
            if vertex.parent == None:
                pass
            else:
                plt.plot([vertex.config[0], vertex.parent.config[0]], [vertex.config[1], vertex.parent.config[1]], color="darkgray")

        for vertex in self.treeVertexGoal[1]:
            if vertex.parent == None:
                pass
            else:
                plt.plot([vertex.config[0], vertex.parent.config[0]], [vertex.config[1], vertex.parent.config[1]], color="darkgray")

        for vertex in self.treeVertexGoal[2]:
            if vertex.parent == None:
                pass
            else:
                plt.plot([vertex.config[0], vertex.parent.config[0]], [vertex.config[1], vertex.parent.config[1]], color="darkgray")

        if True:  # for obstacle space
            jointRange = np.linspace(-np.pi, np.pi, 360)
            xy_points = []
            for theta1 in jointRange:
                for theta2 in jointRange:
                    config = Node(np.array([theta1, theta2]).reshape(2, 1))
                    result = self.robotEnv.collision_check(config)
                    if result is True:
                        xy_points.append([theta1, theta2])

            xy_points = np.array(xy_points)
            plt.plot(xy_points[:, 0], xy_points[:, 1], color='darkcyan', linewidth=0, marker='o', markerfacecolor='darkcyan', markersize=1.5)

        if path1:
            plt.plot([node.config[0] for node in path1], [node.config[1] for node in path1], color='blue', linewidth=2, marker='o', markerfacecolor='plum', markersize=5)

        if path2:
            plt.plot([node.config[0] for node in path2], [node.config[1] for node in path2], color='blue', linewidth=2, marker='o', markerfacecolor='plum', markersize=5)

        if path3:
            plt.plot([node.config[0] for node in path3], [node.config[1] for node in path3], color='blue', linewidth=2, marker='o', markerfacecolor='plum', markersize=5)

        plt.plot(self.xStart.config[0], self.xStart.config[1], color="blue", linewidth=0, marker = 'o', markerfacecolor = 'yellow')

        plt.plot(self.xApp1.config[0], self.xApp1.config[1], color="blue", linewidth=0, marker = 'o', markerfacecolor = 'green')
        plt.plot(self.xGoal1.config[0], self.xGoal1.config[1], color="blue", linewidth=0, marker = 'o', markerfacecolor = 'red')

        plt.plot(self.xApp2.config[0], self.xApp2.config[1], color="blue", linewidth=0, marker = 'o', markerfacecolor = 'green')
        plt.plot(self.xGoal2.config[0], self.xGoal2.config[1], color="blue", linewidth=0, marker = 'o', markerfacecolor = 'red')

        plt.plot(self.xApp3.config[0], self.xApp3.config[1], color="blue", linewidth=0, marker = 'o', markerfacecolor = 'green')
        plt.plot(self.xGoal3.config[0], self.xGoal3.config[1], color="blue", linewidth=0, marker = 'o', markerfacecolor = 'red')

if __name__ == "__main__":
    np.random.seed(9)
    import matplotlib.pyplot as plt
    # import seaborn as sns
    # sns.set_theme()
    # sns.set_context("paper")
    from planner_util.coord_transform import circle_plt
    from util.general_util import write_dict_to_file

    xStart = np.array([0, 0]).reshape(2, 1)

    xApp = [np.array([np.pi/2-0.1, 0.2]).reshape(2, 1),
            np.array([1.45, -0.191]).reshape(2, 1),
            np.array([1.73, -0.160]).reshape(2, 1)]
    
    xGoal = [np.array([np.pi/2, 0]).reshape(2, 1), 
             np.array([np.pi/2, 0]).reshape(2, 1),
             np.array([np.pi/2, 0]).reshape(2, 1)]

    planner = RRTInformedStarDev2D(xStart, xApp, xGoal, eta=0.3, maxIteration=2000)
    planner.robotEnv.robot.plot_arm(xStart, plt_basis=True)
    planner.robotEnv.robot.plot_arm(xGoal[0])
    planner.robotEnv.robot.plot_arm(xGoal[1])
    planner.robotEnv.robot.plot_arm(xGoal[2])
    planner.robotEnv.robot.plot_arm(xApp[0])
    planner.robotEnv.robot.plot_arm(xApp[1])
    planner.robotEnv.robot.plot_arm(xApp[2])
    for obs in planner.robotEnv.taskMapObs:
        obs.plot()
    plt.show()

    path1, path2, path3 = planner.planning()
    print(planner.perfMatrix)
    # write_dict_to_file(planner.perfMatrix, "./planner_dev/result_2d/result_2d_proposed_2000.txt")
    fig, ax = plt.subplots()
    fig.set_size_inches(w=3.40067, h=3.40067)
    fig.tight_layout()
    circle_plt(planner.xGoal1.config[0, 0], planner.xGoal1.config[1, 0], planner.distGoalToAppList[0])
    circle_plt(planner.xGoal2.config[0, 0], planner.xGoal2.config[1, 0], planner.distGoalToAppList[1])
    circle_plt(planner.xGoal3.config[0, 0], planner.xGoal3.config[1, 0], planner.distGoalToAppList[2])
    plt.xlim((-np.pi, np.pi))
    plt.ylim((-np.pi, np.pi))
    planner.plot_tree(path1, path2, path3)
    plt.show()