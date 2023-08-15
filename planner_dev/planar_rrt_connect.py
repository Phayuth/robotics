""" 
Devplanner with RRT connect with reject goal sampling

"""
import os
import sys
wd = os.path.abspath(os.getcwd())
sys.path.append(str(wd))

import time
import numpy as np
from planner_dev.planar_rrt_component import Node, RRTComponent2D


class RRTConnectDev2D(RRTComponent2D):
    def __init__(self, xStart, xApp, xGoal, eta=0.3, maxIteration=1000) -> None:
        super().__init__()
        # start, aux, goal node
        self.xStart = Node(xStart[0, 0], xStart[1, 0])
        self.xGoal = Node(xGoal[0, 0], xGoal[1, 0])
        self.xApp = Node(xApp[0, 0], xApp[1, 0])

        self.treeVertexStart = [self.xStart]
        self.treeVertexGoal = [self.xApp]
        self.probabilityGoalBias = 0.2
        
        self.treeSwapFlag = True
        self.connectNodeStart = None
        self.connectNodeGoal = None
        self.distGoalToApp = self.distance_between_config(self.xGoal, self.xApp)
        self.maxIteration = maxIteration
        self.eta = eta

    def planning(self):
        timePlanningStart = time.perf_counter_ns()
        itera = self.planner_rrt_connect_app()
        timePlanningEnd = time.perf_counter_ns()

        # record performance
        self.perfMatrix["Number of Node"] = len(self.treeVertexStart) + len(self.treeVertexGoal)
        self.perfMatrix["Total Planning Time"] = (timePlanningEnd-timePlanningStart) * 1e-9
        self.perfMatrix["Planning Time Only"] = self.perfMatrix["Total Planning Time"] - self.perfMatrix["KCD Time Spend"]
        self.perfMatrix["Average KCD Time"] = self.perfMatrix["KCD Time Spend"] / self.perfMatrix["Number of Collision Check"]
        
        path = self.search_bidirectional_path()

        return path

    def planner_rrt_connect_app(self):  # Method of Expanding toward Each Other (RRT Connect) + approach pose
        for itera in range(self.maxIteration):
            print(itera)
            if self.treeSwapFlag is True: # Init tree side
                xRand = self.uni_sampling()
                xNearest = self.nearest_node(self.treeVertexStart, xRand)
                xNew = self.steer(xNearest, xRand, self.eta)
                xNew.parent = xNearest

                if not self.is_config_in_collision(xNew) and \
                    not self.is_connect_config_in_collision(xNew.parent, xNew) and \
                    not self.is_config_in_region_of_config(xNew, self.xGoal, self.distGoalToApp) and \
                    not self.is_connect_config_in_region_of_config(xNew.parent, xNew, self.xGoal, self.distGoalToApp):

                    self.treeVertexStart.append(xNew)
                    xNearestPrime = self.nearest_node(self.treeVertexGoal, xNew)
                    xNewPrime = self.steer(xNearestPrime, xNew, self.eta)
                    xNewPrime.parent = xNearestPrime

                    if not self.is_config_in_collision(xNewPrime) and \
                        not self.is_connect_config_in_collision(xNewPrime.parent, xNewPrime)and \
                        not self.is_config_in_region_of_config(xNewPrime, self.xGoal, self.distGoalToApp) and \
                        not self.is_connect_config_in_region_of_config(xNewPrime.parent, xNewPrime, self.xGoal, self.distGoalToApp):

                        self.treeVertexGoal.append(xNewPrime)

                        while True:
                            xNewPPrime = self.steer(xNewPrime, xNew, self.eta)
                            xNewPPrime.parent = xNewPrime

                            # if the 2 node meet, then break
                            if self.distance_between_config(xNewPPrime, xNew) < 1e-3:
                                self.connectNodeGoal = xNewPPrime
                                self.connectNodeStart = xNew
                                break

                            # if there is collision then break and if the node and connection of node to parent is inside region
                            if self.is_config_in_collision(xNewPPrime) or \
                                self.is_connect_config_in_collision(xNewPPrime.parent, xNewPPrime)or \
                                self.is_config_in_region_of_config(xNewPPrime, self.xGoal, self.distGoalToApp) or \
                                self.is_connect_config_in_region_of_config(xNewPPrime.parent, xNewPPrime, self.xGoal, self.distGoalToApp):
                                break

                            # if not collision then free to add
                            else:
                                self.treeVertexGoal.append(xNewPPrime)
                                # oh--! we have to update the xNewPrime to xNewPPrime
                                xNewPrime = xNewPPrime

                if self.connectNodeGoal is not None and self.connectNodeStart is not None:
                    break

                self.tree_swap_flag()

            elif self.treeSwapFlag is False: # App tree side
                xRand = self.uni_sampling()
                xNearest = self.nearest_node(self.treeVertexGoal, xRand)
                xNew = self.steer(xNearest, xRand, self.eta)
                xNew.parent = xNearest

                if not self.is_config_in_collision(xNew) and \
                    not self.is_connect_config_in_collision(xNew.parent, xNew)and \
                    not self.is_config_in_region_of_config(xNew, self.xGoal, self.distGoalToApp) and \
                    not self.is_connect_config_in_region_of_config(xNew.parent, xNew, self.xGoal, self.distGoalToApp):

                    self.treeVertexGoal.append(xNew)
                    xNearestPrime = self.nearest_node(self.treeVertexStart, xNew)
                    xNewPrime = self.steer(xNearestPrime, xNew, self.eta)
                    xNewPrime.parent = xNearestPrime

                    if not self.is_config_in_collision(xNewPrime) and \
                        not self.is_connect_config_in_collision(xNewPrime.parent, xNewPrime)and \
                        not self.is_config_in_region_of_config(xNewPrime, self.xGoal, self.distGoalToApp) and \
                        not self.is_connect_config_in_region_of_config(xNewPrime.parent, xNewPrime, self.xGoal, self.distGoalToApp):

                        self.treeVertexStart.append(xNewPrime)
                        while True:
                            xNewPPrime = self.steer(xNewPrime, xNew, self.eta)
                            xNewPPrime.parent = xNewPrime

                            # if the 2 node meet, then break
                            if self.distance_between_config(xNewPPrime, xNew) < 1e-3:
                                self.connectNodeGoal = xNew
                                self.connectNodeStart = xNewPPrime
                                break

                            # if there is collision then break and if the node and connection of node to parent is inside region
                            if self.is_config_in_collision(xNewPPrime) or \
                                self.is_connect_config_in_collision(xNewPPrime.parent, xNewPPrime) or \
                                self.is_config_in_region_of_config(xNewPPrime, self.xGoal, self.distGoalToApp) or \
                                self.is_connect_config_in_region_of_config(xNewPPrime.parent, xNewPPrime, self.xGoal, self.distGoalToApp):
                                break

                            # if not collision then free to add
                            else:
                                self.treeVertexStart.append(xNewPPrime)
                                # oh--! we have to update the xNewPrime to xNewPPrime
                                xNewPrime = xNewPPrime

                if self.connectNodeGoal is not None and self.connectNodeStart is not None:
                    break

                self.tree_swap_flag()

        return itera

    def search_bidirectional_path(self): # return path is [xinit, x1, x2, ..., xapp, xgoal]
        starterNode = self.connectNodeStart # nearStart
        goalerNode = self.connectNodeGoal   # nearGoal

        pathStart = [starterNode]
        currentNodeStart = starterNode
        while currentNodeStart.parent is not None:
            currentNodeStart = currentNodeStart.parent
            pathStart.append(currentNodeStart)

        pathGoal = [goalerNode]
        currentNodeGoal = goalerNode
        while currentNodeGoal.parent is not None:
            currentNodeGoal = currentNodeGoal.parent
            pathGoal.append(currentNodeGoal)

        pathStart.reverse()
        path = pathStart + pathGoal + [self.xGoal]

        return path
    
    def tree_swap_flag(self):
        if self.treeSwapFlag is True:
            self.treeSwapFlag = False
        elif self.treeSwapFlag is False:
            self.treeSwapFlag = True
    
    def plot_tree(self, path):
        tree = self.treeVertexGoal + self.treeVertexStart
        for vertex in tree:
            plt.scatter(vertex.x, vertex.y, color="sandybrown")
            if vertex.parent == None:
                pass
            else:
                plt.plot([vertex.x, vertex.parent.x], [vertex.y, vertex.parent.y], color="green")

        if path:
                plt.plot([node.x for node in path], [node.y for node in path], color='blue')

        plt.scatter(self.xStart.x, self.xStart.y, color="red")
        plt.scatter(self.xApp.x, self.xApp.y, color="blue")
        plt.scatter(self.xGoal.x, self.xGoal.y, color="yellow")

if __name__ == "__main__":
    np.random.seed(9)
    import matplotlib.pyplot as plt
    # plt.style.use("seaborn")
    from robot.planar_rr import PlanarRR
    from map.taskmap_geo_format import task_rectangle_obs_1
    from planner_util.extract_path_class import extract_path_class_2d
    # from planner_util.plot_util import plot_tree
    from planner_util.coord_transform import circle_plt

    robot = PlanarRR()
    taskMapObs = task_rectangle_obs_1()

    xStart = np.array([0, 0]).reshape(2, 1)
    xGoal = np.array([np.pi/2, 0]).reshape(2, 1)
    xApp = np.array([np.pi/2-0.1, 0.2]).reshape(2, 1)
    robot.plot_arm(xStart, plt_basis=True)
    robot.plot_arm(xGoal)
    robot.plot_arm(xApp)
    for obs in taskMapObs:
        obs.plot()
    plt.show()

    planner = RRTConnectDev2D(xStart, xApp, xGoal, eta=0.3, maxIteration=1000)
    path = planner.planning()
    circle_plt(planner.xGoal.x, planner.xGoal.y, planner.distGoalToApp)
    plt.xlim((-np.pi, np.pi))
    plt.ylim((-np.pi, np.pi))
    planner.plot_tree(path)
    plt.show()

    pathx, pathy = extract_path_class_2d(path)

    plt.axes().set_aspect('equal')
    plt.axvline(x=0, c="green")
    plt.axhline(y=0, c="green")
    obs_list = task_rectangle_obs_1()
    for obs in obs_list:
        obs.plot()
    for i in range(len(path)):
        robot.plot_arm(np.array([[pathx[i]], [pathy[i]]]))
        plt.pause(1)
    plt.show()

    # plot joint 
    t = np.linspace(0,5,len(pathx))
    fig1 = plt.figure("Joint 1")
    ax1 = fig1.add_subplot(111)
    ax1.plot(t,pathx, "ro")
    fig2 = plt.figure("Joint 2")
    ax2 = fig2.add_subplot(111)
    ax2.plot(t,pathy,"ro")
    plt.show()
