""" 
Path Planning Development for UR5 and UR5e   robot with coppeliasim

"""
import os
import sys
wd = os.path.abspath(os.getcwd())
sys.path.append(str(wd))

import time
import numpy as np
from copsim_rrt_component import Node, CopSimRRTComponent


class RRTConnectDev(CopSimRRTComponent):
    def __init__(self, xStart, xApp, xGoal, eta=0.3, maxIteration=1000) -> None:
        super().__init__()
        # start, aux, goal node
        self.xStart = Node(xStart[0, 0], xStart[1, 0], xStart[2, 0], xStart[3, 0], xStart[4, 0], xStart[5, 0])
        self.xGoal = Node(xGoal[0, 0], xGoal[1, 0], xGoal[2, 0], xGoal[3, 0], xGoal[4, 0], xGoal[5, 0])
        self.xApp = Node(xApp[0, 0], xApp[1, 0], xApp[2, 0], xApp[3, 0], xApp[4, 0], xApp[5, 0])

        self.eta = eta
        self.maxIteration = maxIteration
        self.treeVertexStart = [self.xStart]
        self.treeVertexGoal = [self.xApp]
        self.treeSwapFlag = True
        self.connectNodeStart = None
        self.connectNodeGoal = None
        self.distGoalToApp = self.distance_between_config(self.xGoal, self.xApp)

    def planning(self):
        self.copHandle.start_sim()
        # planning stage
        timePlanningStart = time.perf_counter_ns()
        # itera = self.planner_generic_bidirectional()
        # itera = self.planner_rrt_connect()
        itera = self.planner_rrt_connect_app()
        timePlanningEnd = time.perf_counter_ns()
        print("Finished Tree Building")

        # search path stage
        timeSearchStart = time.perf_counter_ns()
        path = self.search_bidirectional_path()
        timeSearchEnd = time.perf_counter_ns()
        print("Finshed Path Search")

        # optimization stage
        timeOptStart = time.perf_counter_ns()
        pathPruned = None
        # pathPruned = self.optimizer_greedy_prune_path(path)
        # pathPruned = self.optimizer_mid_node_prune_path((path))
        pathToApproch = None
        # pathToApproch = self.optimizer_apply_approach_path(path)
        # pathSeg = self.optimizer_segment_linear_interpolation_path(pathPruned)
        timeOptEnd = time.perf_counter_ns()
        print("Finshed Path Optimzation")

        # record performance
        self.perfMatrix["totalPlanningTime"] = (timePlanningEnd-timePlanningStart) * 1e-9
        self.perfMatrix["KCDTimeSpend"] = self.perfMatrix["KCDTimeSpend"] * 1e-9
        self.perfMatrix["planningTimeOnly"] = self.perfMatrix["totalPlanningTime"] - self.perfMatrix["KCDTimeSpend"]
        self.perfMatrix["numberOfKCD"] = len(self.configSearched)
        self.perfMatrix["avgKCDTime"] = self.perfMatrix["KCDTimeSpend"] * 1e-9 / self.perfMatrix["numberOfKCD"]
        self.perfMatrix["numberOfNodeTreeStart"] = len(self.treeVertexStart)
        self.perfMatrix["numberOfNodeTreeGoal"] = len(self.treeVertexGoal)
        self.perfMatrix["numberOfNode"] = len(self.treeVertexGoal) + len(self.treeVertexStart)
        self.perfMatrix["numberOfMaxIteration"] = self.maxIteration
        self.perfMatrix["numberOfIterationUsed"] = itera + 1
        self.perfMatrix["searchPathTime"] = (timeSearchEnd-timeSearchStart) * 1e-9
        self.perfMatrix["numberOfOriginalPath"] = len(path)
        if pathPruned is not None:
            self.perfMatrix["numberOfPathPruned"] = len(pathPruned)
            self.perfMatrix["optimizationTime"] = (timeOptEnd-timeOptStart) * 1e-9
        if pathToApproch is not None:
            self.perfMatrix["numberOfPathToApproach"] = len(pathToApproch)

        self.copHandle.stop_sim()
        return path
        # return pathPruned
        # return pathToApproch
        # return pathSeg

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
                        not self.is_connect_config_in_collision(xNewPrime.parent, xNewPrime) and \
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
                                self.is_connect_config_in_collision(xNewPPrime.parent, xNewPPrime) or \
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
                    not self.is_connect_config_in_collision(xNew.parent, xNew) and \
                    not self.is_config_in_region_of_config(xNew, self.xGoal, self.distGoalToApp) and \
                    not self.is_connect_config_in_region_of_config(xNew.parent, xNew, self.xGoal, self.distGoalToApp):

                    self.treeVertexGoal.append(xNew)
                    xNearestPrime = self.nearest_node(self.treeVertexStart, xNew)
                    xNewPrime = self.steer(xNearestPrime, xNew, self.eta)
                    xNewPrime.parent = xNearestPrime

                    if not self.is_config_in_collision(xNewPrime) and \
                        not self.is_connect_config_in_collision(xNewPrime.parent, xNewPrime) and \
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
        # nearStart, nearGoal = self.is_both_tree_node_near(return_near_node=True)
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

    def is_both_tree_node_near(self, return_near_node=False):
        for eachVertexStart in self.treeVertexStart:
            for eachVertexGoal in self.treeVertexGoal:
                if self.distance_between_config(eachVertexStart, eachVertexGoal) <= self.eta:
                    if return_near_node:
                        return eachVertexStart, eachVertexGoal
                    return True
        return False

if __name__ == "__main__":
    np.random.seed(9)
    import matplotlib.pyplot as plt
    from scipy.optimize import curve_fit
    from planner_util.extract_path_class import extract_path_class_6d
    from util.dictionary_pretty import print_dict
    from target_localization.pre_record_value import thetaInit, thetaGoal4, thetaApp4

    # Define pose
    thetaInit = thetaInit
    thetaGoal = thetaGoal4
    thetaApp = thetaApp4
    
    planner = RRTConnectDev(thetaInit, thetaApp, thetaGoal, eta=0.1, maxIteration=5000)
    path = planner.planning()
    # print(f"==>> path: \n{path}")
    print_dict(planner.perfMatrix)

    time.sleep(3)

    # play back
    planner.copHandle.start_sim()
    pathX, pathY, pathZ, pathP, pathQ, pathR = extract_path_class_6d(path)
    pathX = np.array(pathX)
    pathY = np.array(pathY)
    pathZ = np.array(pathZ)
    pathP = np.array(pathP)
    pathQ = np.array(pathQ)
    pathR = np.array(pathR)

    # loop in simulation
    for i in range(len(pathX)):
        jointVal = np.array([pathX[i], pathY[i], pathZ[i], pathP[i], pathQ[i], pathR[i]]).reshape(6, 1)
        planner.copHandle.set_joint_value(jointVal)
        time.sleep(0.5)
        # triggers next simulation step
        # client.step()

    # stop simulation
    planner.copHandle.stop_sim()

    # Optimization stage, I want to fit the current theta to time and use that information to inform optimization
    time = np.linspace(0, 1, len(pathX))

    def quintic5deg(x, a, b, c, d, e, f):
        return a * x**5 + b * x**4 + c * x**3 + d * x**2 + e*x*f
    
    def polynomial9deg(x, a, b, c, d, e, f, g, h, i):
        return a*x**9 + b*x**8 + c*x**7 + d*x**6 + e*x**5 + f*x**4 + g*x**3 + h*x**2 + i*x

    # force start and end point to fit  https://stackoverflow.com/questions/33539287/how-to-force-specific-points-in-curve-fitting
    sigma = np.ones(len(pathX))
    sigma[[0, -1]] = 0.01

    # Fit the line equation
    # poptX, pcovX = curve_fit(quintic5deg, time, pathX, sigma=sigma)
    # poptY, pcovY = curve_fit(quintic5deg, time, pathY, sigma=sigma)
    # poptZ, pcovZ = curve_fit(quintic5deg, time, pathZ, sigma=sigma)
    # poptP, pcovP = curve_fit(quintic5deg, time, pathP, sigma=sigma)
    # poptQ, pcovQ = curve_fit(quintic5deg, time, pathQ, sigma=sigma)
    # poptR, pcovR = curve_fit(quintic5deg, time, pathR, sigma=sigma)
    # quintic5deg(timeSmooth[i], *poptX),
    timeSmooth = np.linspace(0, 1, 100)

    fig4, axes = plt.subplots(6, 1, sharex='all')
    axes[0].plot(time, pathX, 'ro')
    # axes[0].plot(time, quintic5deg(time, *poptX))
    axes[1].plot(time, pathY, 'ro')
    # axes[1].plot(time, quintic5deg(time, *poptY))
    axes[2].plot(time, pathZ, 'ro')
    # axes[2].plot(time, quintic5deg(time, *poptZ))
    axes[3].plot(time, pathP, 'ro')
    # axes[3].plot(time, quintic5deg(time, *poptP))
    axes[4].plot(time, pathQ, 'ro')
    # axes[4].plot(time, quintic5deg(time, *poptQ))
    axes[5].plot(time, pathR, 'ro')
    # axes[5].plot(time, quintic5deg(time, *poptR))
    plt.show()