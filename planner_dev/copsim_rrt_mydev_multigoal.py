""" 
Path Planning Development for UR5 and UR5e robot with coppeliasim 
RRT Connect for Initial Path
RRT Informed for Improvement
Multiple Goal Point

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

        self.xStart = Node(xStart[0, 0], xStart[1, 0], xStart[2, 0], xStart[3, 0], xStart[4, 0], xStart[5, 0])

        self.xGoal = Node(xGoal[0, 0], xGoal[1, 0], xGoal[2, 0], xGoal[3, 0], xGoal[4, 0], xGoal[5, 0])
        self.xApp = Node(xApp[0, 0], xApp[1, 0], xApp[2, 0], xApp[3, 0], xApp[4, 0], xApp[5, 0])

        self.xGoal = Node(xGoal[0, 0], xGoal[1, 0], xGoal[2, 0], xGoal[3, 0], xGoal[4, 0], xGoal[5, 0])
        self.xApp = Node(xApp[0, 0], xApp[1, 0], xApp[2, 0], xApp[3, 0], xApp[4, 0], xApp[5, 0])

        self.xGoal = Node(xGoal[0, 0], xGoal[1, 0], xGoal[2, 0], xGoal[3, 0], xGoal[4, 0], xGoal[5, 0])
        self.xApp = Node(xApp[0, 0], xApp[1, 0], xApp[2, 0], xApp[3, 0], xApp[4, 0], xApp[5, 0])

        self.treeVertexStart = [self.xStart]
        self.treeVertexGoal = [self.xApp]

        self.eta = eta
        self.maxIteration = maxIteration

        self.treeSwapFlag = True
        self.connectNodeStart = None
        self.connectNodeGoal = None
        self.rewireRadius = 1
        self.foundInitialPath = False
        self.distGoalToApp = self.distance_between_config(self.xGoal, self.xApp)
        self.XSoln = []
        self.mergeTree = False
        self.path1 = []
        self.path2 = []
        self.path3 = []

    def planning(self):
        self.copHandle.start_sim()

        timePlanningStart = time.perf_counter_ns()
        itera = self.planner_rrt_connect_informed()
        path = self.search_singledirection_path()
        timePlanningEnd = time.perf_counter_ns()

        # record performance
        self.perfMatrix["Number of Node"] = len(self.treeVertexStart)
        self.perfMatrix["Total Planning Time"] = (timePlanningEnd-timePlanningStart) * 1e-9
        self.perfMatrix["KCD Time Spend"] = self.perfMatrix["KCD Time Spend"] * 1e-9
        self.perfMatrix["Planning Time Only"] = self.perfMatrix["Total Planning Time"] - self.perfMatrix["KCD Time Spend"]
        self.perfMatrix["Average KCD Time"] = self.perfMatrix["KCD Time Spend"] / self.perfMatrix["Number of Collision Check"]

        self.copHandle.stop_sim()

        return path

    def planner_rrt_connect_informed(self):
        for itera in range(self.maxIteration):
            print(itera)
            if self.foundInitialPath is False:
                if self.treeSwapFlag is True: # Init tree side
                    xRand = self.uni_sampling()
                    xNearest = self.nearest_node(self.treeVertexStart, xRand)
                    xNew = self.steer(xNearest, xRand, self.eta)
                    xNew.parent = xNearest
                    xNew.cost = xNew.parent.cost + self.cost_line(xNew, xNew.parent)
                    
                    if not self.is_config_in_collision(xNew) and \
                        not self.is_connect_config_in_collision(xNew.parent, xNew) and \
                        not self.is_config_in_region_of_config(xNew, self.xGoal, self.distGoalToApp) and \
                        not self.is_connect_config_in_region_of_config(xNew.parent, xNew, self.xGoal, self.distGoalToApp):

                        self.treeVertexStart.append(xNew)
                        xNearestPrime = self.nearest_node(self.treeVertexGoal, xNew)
                        xNewPrime = self.steer(xNearestPrime, xNew, self.eta)
                        xNewPrime.parent = xNearestPrime
                        xNewPrime.cost = xNewPrime.parent.cost + self.cost_line(xNewPrime, xNewPrime.parent)

                        if not self.is_config_in_collision(xNewPrime) and \
                            not self.is_connect_config_in_collision(xNewPrime.parent, xNewPrime) and \
                            not self.is_config_in_region_of_config(xNewPrime, self.xGoal, self.distGoalToApp) and \
                            not self.is_connect_config_in_region_of_config(xNewPrime.parent, xNewPrime, self.xGoal, self.distGoalToApp):

                            self.treeVertexGoal.append(xNewPrime)

                            while True:
                                xNewPPrime = self.steer(xNewPrime, xNew, self.eta)
                                xNewPPrime.parent = xNewPrime
                                xNewPPrime.cost = xNewPPrime.parent.cost + self.cost_line(xNewPPrime, xNewPPrime.parent)
                                
                                # if the 2 node meet, then break
                                if self.distance_between_config(xNewPPrime, xNew) < 1e-3:
                                    self.connectNodeGoal = xNewPPrime
                                    self.connectNodeStart = xNew
                                    self.foundInitialPath = True
                                    break # break out sampling and break out of planner 

                                # if there is collision then break and if the node and connection of node to parent is inside region
                                if self.is_config_in_collision(xNewPPrime) or \
                                    self.is_connect_config_in_collision(xNewPPrime.parent, xNewPPrime) or \
                                    self.is_config_in_region_of_config(xNewPPrime, self.xGoal, self.distGoalToApp) or \
                                    self.is_connect_config_in_region_of_config(xNewPPrime.parent, xNewPPrime, self.xGoal, self.distGoalToApp):
                                    break

                                # if not collision then free to add
                                else:
                                    self.treeVertexGoal.append(xNewPPrime)
                                    xNewPrime = xNewPPrime

                    if self.connectNodeGoal is not None and self.connectNodeStart is not None:
                        self.reparent_merge_tree()
                        # break

                    self.tree_swap_flag()

                elif self.treeSwapFlag is False: # App tree side
                    xRand = self.uni_sampling()
                    xNearest = self.nearest_node(self.treeVertexGoal, xRand)
                    xNew = self.steer(xNearest, xRand, self.eta)
                    xNew.parent = xNearest
                    xNew.cost = xNew.parent.cost + self.cost_line(xNew, xNew.parent)

                    if not self.is_config_in_collision(xNew) and \
                        not self.is_connect_config_in_collision(xNew.parent, xNew) and \
                        not self.is_config_in_region_of_config(xNew, self.xGoal, self.distGoalToApp) and \
                        not self.is_connect_config_in_region_of_config(xNew.parent, xNew, self.xGoal, self.distGoalToApp):

                        self.treeVertexGoal.append(xNew)
                        xNearestPrime = self.nearest_node(self.treeVertexStart, xNew)
                        xNewPrime = self.steer(xNearestPrime, xNew, self.eta)
                        xNewPrime.parent = xNearestPrime
                        xNewPrime.cost = xNewPrime.parent.cost + self.cost_line(xNewPrime, xNewPrime.parent)

                        if not self.is_config_in_collision(xNewPrime) and \
                            not self.is_connect_config_in_collision(xNewPrime.parent, xNewPrime) and \
                            not self.is_config_in_region_of_config(xNewPrime, self.xGoal, self.distGoalToApp) and \
                            not self.is_connect_config_in_region_of_config(xNewPrime.parent, xNewPrime, self.xGoal, self.distGoalToApp):

                            self.treeVertexStart.append(xNewPrime)
                            while True:
                                xNewPPrime = self.steer(xNewPrime, xNew, self.eta)
                                xNewPPrime.parent = xNewPrime
                                xNewPPrime.cost = xNewPPrime.parent.cost + self.cost_line(xNewPPrime, xNewPPrime.parent)

                                # if the 2 node meet, then break
                                if self.distance_between_config(xNewPPrime, xNew) < 1e-3:
                                    self.connectNodeGoal = xNew
                                    self.connectNodeStart = xNewPPrime
                                    self.foundInitialPath = True
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
                                    xNewPrime = xNewPPrime

                    if self.connectNodeGoal is not None and self.connectNodeStart is not None:
                        self.reparent_merge_tree()
                        # break

                    self.tree_swap_flag()

            else:
                print("Enter Informed Mode")
                if len(self.XSoln) == 0:
                    cBest = self.xApp.cost
                else:
                    xSolnCost = [xSoln.parent.cost + self.cost_line(xSoln.parent, xSoln) + self.cost_line(xSoln, self.xApp) for xSoln in self.XSoln]
                    xSolnCost = xSolnCost + [self.xApp.cost]
                    print(f"==>> xSolnCost: \n{xSolnCost}")
                    cBest = min(xSolnCost)

                xRand = self.informed_sampling(self.xStart, self.xApp, cBest)
                xNearest = self.nearest_node(self.treeVertexStart, xRand)
                xNew = self.steer(xNearest, xRand, self.eta)
                xNew.parent = xNearest
                if self.is_config_in_region_of_config(xNew, self.xGoal, self.distGoalToApp) or \
                    self.is_connect_config_in_region_of_config(xNew.parent, xNew, self.xGoal, self.distGoalToApp):
                    continue
                xNew.cost = xNew.parent.cost + self.cost_line(xNew, xNew.parent)
                if self.is_config_in_collision(xNew) or self.is_connect_config_in_collision(xNew.parent, xNew):
                    continue
                else:
                    XNear = self.near(self.treeVertexStart, xNew, self.rewireRadius)
                    xMin = xNew.parent
                    cMin = xMin.cost + self.cost_line(xMin, xNew)
                    for xNear in XNear:
                        if self.is_connect_config_in_collision(xNear, xNew):
                            continue

                        cNew = xNear.cost + self.cost_line(xNear, xNew)
                        if cNew < cMin:
                            xMin = xNear
                            cMin = cNew

                    xNew.parent = xMin
                    xNew.cost = cMin
                    self.treeVertexStart.append(xNew)

                    for xNear in XNear:
                        if self.is_connect_config_in_collision(xNear, xNew):
                            continue
                        cNear = xNear.cost
                        cNew = xNew.cost + self.cost_line(xNew, xNear)
                        if cNew < cNear:
                            xNear.parent = xNew
                            xNear.cost = xNew.cost + self.cost_line(xNew, xNear)

                    # in approach region
                    if self.is_config_in_region_of_config(xNew, self.xApp, radius=1):
                        self.XSoln.append(xNew)

        return itera

    def tree_swap_flag(self):
        if self.treeSwapFlag is True:
            self.treeSwapFlag = False
        elif self.treeSwapFlag is False:
            self.treeSwapFlag = True

    def reparent_merge_tree(self): #have to consider parent and cost re update and have to update all node in tree goal as well
        xTobeParent = self.connectNodeStart
        xNow = self.connectNodeGoal
        while True:
            if xNow.parent is None: # xnow is xApp cause it has no parent, so we arrive at xApp, so we update its parent, cost then kill the process
                xNow.parent = xTobeParent
                xNow.cost = xNow.parent.cost + self.cost_line(xNow.parent, xNow)
                self.treeVertexStart.append(xNow)
                break
            xParentSave = xNow.parent
            xNow.parent = xTobeParent
            xNow.cost = xNow.parent.cost + self.cost_line(xNow.parent, xNow)
            self.treeVertexStart.append(xNow)
            xTobeParent = xNow
            xNow = xParentSave

        # for x in self.treeVertexGoal: # update cost of node in treeGoal but we have to update from the main branch first
        #     x.cost = x.parent.cost + self.cost_line(x.parent, x) # not correct yet

    # def search_singledirection_path(self):
    #     XNear = self.near(self.treeVertexStart, self.xApp, self.rewireRadius)
    #     for xNear in XNear:
    #         if self.is_connect_config_in_collision(xNear, self.xApp):
    #             continue
    #         self.xApp.parent = xNear

    #         path = [self.xApp]
    #         currentNode = self.xApp

    #         while currentNode != self.xStart:
    #             currentNode = currentNode.parent
    #             path.append(currentNode)

    #         path.reverse()
    #         bestPath = path
    #         cost = sum(i.cost for i in path)

    #         if cost < sum(j.cost for j in bestPath):
    #             bestPath = path

    #     return bestPath + [self.xGoal]

    def search_singledirection_path(self):
        vertexList = []
        # for xNear in self.treeVertexStart:
        for xNear in self.XSoln:
            if self.is_connect_config_in_collision(xNear, self.xApp):
                continue
            self.xApp.parent = xNear

            path = [self.xApp]
            currentNodeStart = self.xApp
            while currentNodeStart.parent is not None:
                currentNodeStart = currentNodeStart.parent
                path.append(currentNodeStart)

            cost = sum(i.cost for i in path)
            vertexList.append(cost)
        
        minIndex = np.argmin(vertexList)
        xBest = self.XSoln[minIndex]

        self.xApp.parent = xBest
        bestPath = [self.xApp]
        currentNodeStart = self.xApp
        while currentNodeStart.parent is not None:
            currentNodeStart = currentNodeStart.parent
            bestPath.append(currentNodeStart)

        bestPath.reverse()

        return bestPath + [self.xGoal]
    
    def search_singledirection_nocost_path(self):
        starterNode = self.xApp

        pathStart = [starterNode]
        currentNodeStart = starterNode
        while currentNodeStart.parent is not None:
            currentNodeStart = currentNodeStart.parent
            pathStart.append(currentNodeStart)

        pathStart.reverse()

        return pathStart

    def informed_sampling(self, xStart, xGoal, cMax):
        cMin = self.distance_between_config(xStart, xGoal)
        print(cMax, cMin)
        xCenter = np.array([(xStart.x + xGoal.x) / 2,
                            (xStart.y + xGoal.y) / 2,
                            (xStart.z + xGoal.z) / 2,
                            (xStart.p + xGoal.p) / 2,
                            (xStart.q + xGoal.q) / 2,
                            (xStart.r + xGoal.r) / 2]).reshape(6, 1)

        L, C = self.rotation_to_world(xStart, xGoal, cMax, cMin)

        while True:
            xBall = self.unit_ball_sampling()
            xRand = (C@L@xBall) + xCenter
            xRand = Node(xRand[0, 0], xRand[1, 0], xRand[2, 0], xRand[3, 0], xRand[4, 0], xRand[5, 0])
            in_range = [(self.xMinRange < xRand.x < self.xMaxRange),
                        (self.yMinRange < xRand.y < self.yMaxRange),
                        (self.zMinRange < xRand.z < self.zMaxRange), 
                        (self.pMinRange < xRand.p < self.pMaxRange),
                        (self.qMinRange < xRand.q < self.qMaxRange),
                        (self.rMinRange < xRand.r < self.rMaxRange)]
            if all(in_range):
                break
        return xRand

    def unit_ball_sampling(self):
        u = np.random.normal(0, 1, (1, 6 + 2))
        norm = np.linalg.norm(u, axis = -1, keepdims = True)
        u = u/norm
        return u[0,:6].reshape(6,1) #The first N coordinates are uniform in a unit N ball
    
    def rotation_to_world(self, xStart, xGoal, cMax, cMin):
        r1 = cMax / 2
        r2to6 = np.sqrt(cMax**2 - cMin**2) / 2
        L = np.diag([r1, r2to6, r2to6, r2to6, r2to6, r2to6])
        a1 = np.array([[(xGoal.x - xStart.x) / cMin],
                       [(xGoal.y - xStart.y) / cMin],
                       [(xGoal.z - xStart.z) / cMin],
                       [(xGoal.p - xStart.p) / cMin],
                       [(xGoal.q - xStart.q) / cMin],
                       [(xGoal.r - xStart.r) / cMin]])
        I1 = np.array([[1.0], [0.0], [0.0], [0.0], [0.0], [0.0]])
        M = a1 @ I1.T
        U, _, V_T = np.linalg.svd(M, True, True)
        C = U @ np.diag([1.0, 1.0, 1.0, 1.0, 1.0, np.linalg.det(U) * np.linalg.det(V_T.T)]) @ V_T
        return L, C


if __name__ == "__main__":
    np.random.seed(9)
    import matplotlib.pyplot as plt
    from scipy.optimize import curve_fit
    from planner_util.extract_path_class import extract_path_class_6d
    from util.dictionary_pretty import print_dict
    from target_localization.pre_record_value import thetaInit, thetaGoal4, thetaApp4, qCurrent, qAux, qGoal, wrap_to_pi
    from copsim.arm_api import UR5eStateArmCoppeliaSimAPI

    # Define pose
    thetaInit = wrap_to_pi(thetaInit)
    thetaGoal = wrap_to_pi(qGoal)
    thetaApp = wrap_to_pi(qAux)
    
    planner = RRTConnectDev(thetaInit, thetaApp, thetaGoal, eta=0.3, maxIteration=2000)
    path = planner.planning()
    print(f"==>> path: \n{path}")
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


    armState = UR5eStateArmCoppeliaSimAPI()
    armState.set_goal_joint_value(thetaGoal)
    armState.set_aux_joint_value(thetaApp)
    armState.set_start_joint_value(thetaInit)
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
