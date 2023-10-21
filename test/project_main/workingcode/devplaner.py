""" Path Planning Development for 6 dof robot
- Main:    RRT            |   Added
- Variant: Bi Directional |   Added
- Variant: RRT Connect    |   Added https://github.com/zhm-real/PathPlanning/blob/master/Sampling_based_Planning/rrt_2D/rrt_connect.py
- Goal Rejection          |   Added
- Sampling : Bias         |   Added
- Sampling : Smart        |   Currently Adding
- Pruning                 |   Added
- Multiple Waypoint       |   ?
- Point Informed Resampling Optimization | Currently Adding (More or Less like STOMP optimazation, currently investigate)
- Vector database collision store        | Added but it is not good and slow

"""

import os
import sys

wd = os.path.abspath(os.getcwd())
sys.path.append(str(wd))

import time
import numpy as np
from geometry.geometry_class import ObjLine2D, intersect_line_v_rectangle


class Node:

    def __init__(self, x, y, z, p, q, r, parent=None, cost=0.0) -> None:
        self.x = x
        self.y = y
        self.z = z
        self.p = p
        self.q = q
        self.r = r
        self.parent = parent
        self.cost = cost

    def __repr__(self) -> str:
        return f'\nconfig = [{self.x:.7f}, {self.y:.7f}, {self.z:.7f}, {self.p:.7f}, {self.q:.7f}, {self.r:.7f}, hasParent = {True if self.parent != None else False}]'

class DevPlanner():

    def __init__(self, robot, taskMapObs, xStart, xApp, xGoal, eta=0.3, maxIteration=1000) -> None:
        # robot and workspace
        self.robot = robot
        self.taskMapObs = taskMapObs

        self.xMinRange = 0
        self.xMaxRange = np.pi
        self.yMinRange = -np.pi
        self.yMaxRange = np.pi
        self.zMinRange = -np.pi
        self.zMaxRange = np.pi
        self.pMinRange = -np.pi
        self.pMaxRange = np.pi
        self.qMinRange = -np.pi
        self.qMaxRange = np.pi
        self.rMinRange = -np.pi
        self.rMaxRange = np.pi

        self.probabilityGoalBias = 0.4
        self.xStart = Node(xStart[0, 0], xStart[1, 0], xStart[2, 0], xStart[3, 0], xStart[4, 0], xStart[5, 0])
        self.xGoal = Node(xGoal[0, 0], xGoal[1, 0], xGoal[2, 0], xGoal[3, 0], xGoal[4, 0], xGoal[5, 0])
        self.xApp = Node(xApp[0, 0], xApp[1, 0], xApp[2, 0], xApp[3, 0], xApp[4, 0], xApp[5, 0])

        # properties of planner
        self.maxIteration = maxIteration
        self.eta = eta
        self.treeVertexStart = [self.xStart]
        self.treeVertexGoal = [self.xApp]
        self.treeSwapFlag = True
        self.connectNodeStart = None
        self.connectNodeGoal = None
        self.distGoalToApp = self.distance_between_config(self.xGoal, self.xApp)
        self.nodeGoalSphere = []

        # collision database
        self.configSearched = []
        self.collisionState = []

        # performance matrix
        self.perfMatrix = {
            "totalPlanningTime": 0.0,  # include KCD
            "KCDTimeSpend": 0.0,
            "planningTimeOnly": 0.0,
            "numberOfKCD": 0,
            "avgKCDTime": 0.0,
            "numberOfNodeTreeStart": 0,
            "numberOfNodeTreeGoal" : 0,
            "numberOfNode": 0,
            "numberOfMaxIteration": 0,
            "numberOfIterationUsed": 0,
            "searchPathTime": 0.0,
            "numberOfOriginalPath" : 0,
            "numberOfPathPruned": 0,
            "optimizationTime" : 0.0,
            "numberOfPathToApproach": 0
        }

    def planning(self):
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
        # pathPruned = None
        pathPruned = self.optimizer_greedy_prune_path(path)
        # pathPruned = self.optimizer_mid_node_prune_path((path))
        pathToApproch = None
        # pathToApproch = self.optimizer_apply_approach_path(path)
        pathSeg = self.optimizer_segment_linear_interpolation_path(pathPruned)
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
        self.perfMatrix["numberOfIterationUsed"] = itera
        self.perfMatrix["searchPathTime"] = (timeSearchEnd-timeSearchStart) * 1e-9
        self.perfMatrix["numberOfOriginalPath"] = len(path)
        if pathPruned is not None:
            self.perfMatrix["numberOfPathPruned"] = len(pathPruned)
            self.perfMatrix["optimizationTime"] = (timeOptEnd - timeOptStart) * 1e-9
        if pathToApproch is not None:
            self.perfMatrix["numberOfPathToApproach"] = len(pathToApproch)

        # return path
        # return pathPruned
        # return pathToApproch
        return pathSeg

    def planner_generic_bidirectional(self):  # Method of Expanding toward Random Node (Generic Bidirectional)
        for itera in range(self.maxIteration):
            print(itera)
            xRandStart = self.bias_sampling(self.xApp)
            xRandGoal = self.bias_sampling(self.xStart)
            xNearestStart = self.nearest_node(self.treeVertexStart, xRandStart)
            xNearestGoal = self.nearest_node(self.treeVertexGoal, xRandGoal)
            xNewStart = self.steer(xNearestStart, xRandStart)
            xNewGoal = self.steer(xNearestGoal, xRandGoal)

            # ingoal region rejection
            if self.is_config_in_region_of_config(xNewGoal, self.xGoal, radius=0.4) or self.is_connect_config_in_region_of_config(xNearestStart, xNewStart, self.xGoal, radius=0.4):
                if self.is_config_in_region_of_config(xNewStart, self.xGoal, radius=0.4) or self.is_connect_config_in_region_of_config(xNearestGoal, xNewGoal, self.xGoal, radius=0.4):
                    continue

            if self.is_config_in_collision(xNewStart) or self.is_connect_config_in_collision(xNearestStart, xNewStart):
                continue
            else:
                self.treeVertexStart.append(xNewStart)

            if self.is_config_in_collision(xNewGoal) or self.is_connect_config_in_collision(xNearestGoal, xNewGoal):
                continue
            else:
                self.treeVertexGoal.append(xNewGoal)

        return itera
    
    def planner_rrt_connect(self):  # Method of Expanding toward Each Other (RRT Connect)
        for itera in range(self.maxIteration):
            print(itera)
            # add a sample of surface sphere center at goal and radius equal to distance from xgoal to xapp
            # sphereNode = self.hypersphere_surface_sampling(self.xGoal, self.distance_between_config(self.xGoal, self.xApp))
            # if self.is_config_in_collision(sphereNode):
            #     self.nodeGoalSphere.append(sphereNode)

            if self.treeSwapFlag is True:
                # xRand = self.bias_sampling(self.treeVertexGoal[0])
                xRand = self.uni_sampling()
                xNearest = self.nearest_node(self.treeVertexStart, xRand)
                xNew = self.steer(xNearest, xRand)
                xNew.parent = xNearest

                if not self.is_config_in_collision(xNew) and not self.is_connect_config_in_collision(xNew.parent, xNew):
                    self.treeVertexStart.append(xNew)
                    xNearestPrime = self.nearest_node(self.treeVertexGoal, xNew)
                    xNewPrime = self.steer(xNearestPrime, xNew)
                    xNewPrime.parent = xNearestPrime

                    if not self.is_config_in_collision(xNewPrime) and not self.is_connect_config_in_collision(xNewPrime.parent, xNewPrime):
                        self.treeVertexGoal.append(xNewPrime)

                        while True:
                            xNewPPrime = self.steer(xNewPrime, xNew)
                            xNewPPrime.parent = xNewPrime

                            # if the 2 node meet, then break
                            if self.distance_between_config(xNewPPrime, xNew) < 1e-3: 
                                self.connectNodeGoal = xNewPPrime
                                self.connectNodeStart = xNew
                                break

                            # if there is collision then break
                            if self.is_config_in_collision(xNewPPrime) or self.is_connect_config_in_collision(xNewPPrime.parent, xNewPPrime):
                                break

                            # if not collision then free to add
                            else:
                                self.treeVertexGoal.append(xNewPPrime)
                                # oh--! we have to update the xNewPrime to xNewPPrime
                                xNewPrime = xNewPPrime

                if self.connectNodeGoal is not None and self.connectNodeStart is not None:
                    break

                self.tree_swap_flag() 
                # it can be easily done by switch tree like under, but I want to express a better understanding with swapping tree like that
                # self.treeVertexStart, self.treeVertexGoal = self.treeVertexGoal, self.treeVertexStart
            
            elif self.treeSwapFlag is False:
                # xRand = self.bias_sampling(self.treeVertexStart[0])
                xRand = self.uni_sampling()
                xNearest = self.nearest_node(self.treeVertexGoal, xRand)
                xNew = self.steer(xNearest, xRand)
                xNew.parent = xNearest

                if not self.is_config_in_collision(xNew) and not self.is_connect_config_in_collision(xNew.parent, xNew):
                    self.treeVertexGoal.append(xNew)
                    xNearestPrime = self.nearest_node(self.treeVertexStart, xNew)
                    xNewPrime = self.steer(xNearestPrime, xNew)
                    xNewPrime.parent = xNearestPrime

                    if not self.is_config_in_collision(xNewPrime) and not self.is_connect_config_in_collision(xNewPrime.parent, xNewPrime):
                        self.treeVertexStart.append(xNewPrime)

                        while True:
                            xNewPPrime = self.steer(xNewPrime, xNew)
                            xNewPPrime.parent = xNewPrime

                            # if the 2 node meet, then break
                            if self.distance_between_config(xNewPPrime, xNew) < 1e-3: 
                                self.connectNodeGoal = xNew
                                self.connectNodeStart = xNewPPrime
                                break

                            # if there is collision then break
                            if self.is_config_in_collision(xNewPPrime) or self.is_connect_config_in_collision(xNewPPrime.parent, xNewPPrime):
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
    
    def planner_rrt_connect_app(self):  # Method of Expanding toward Each Other (RRT Connect) + approach pose
        for itera in range(self.maxIteration):
            print(itera)
            if self.treeSwapFlag is True: # Init tree side
                xRand = self.uni_sampling()
                xNearest = self.nearest_node(self.treeVertexStart, xRand)
                xNew = self.steer(xNearest, xRand)
                xNew.parent = xNearest
                    
                if not self.is_config_in_collision(xNew) and \
                    not self.is_connect_config_in_collision(xNew.parent, xNew) and \
                    not self.is_config_in_region_of_config(xNew, self.xGoal, self.distGoalToApp) and \
                    not self.is_connect_config_in_region_of_config(xNew.parent, xNew, self.xGoal, self.distGoalToApp):

                    self.treeVertexStart.append(xNew)
                    xNearestPrime = self.nearest_node(self.treeVertexGoal, xNew)
                    xNewPrime = self.steer(xNearestPrime, xNew)
                    xNewPrime.parent = xNearestPrime

                    if not self.is_config_in_collision(xNewPrime) and \
                        not self.is_connect_config_in_collision(xNewPrime.parent, xNewPrime) and \
                        not self.is_config_in_region_of_config(xNewPrime, self.xGoal, self.distGoalToApp) and \
                        not self.is_connect_config_in_region_of_config(xNewPrime.parent, xNewPrime, self.xGoal, self.distGoalToApp):

                        self.treeVertexGoal.append(xNewPrime)

                        while True:
                            xNewPPrime = self.steer(xNewPrime, xNew)
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
                xNew = self.steer(xNearest, xRand)
                xNew.parent = xNearest

                if not self.is_config_in_collision(xNew) and \
                    not self.is_connect_config_in_collision(xNew.parent, xNew) and \
                    not self.is_config_in_region_of_config(xNew, self.xGoal, self.distGoalToApp) and \
                    not self.is_connect_config_in_region_of_config(xNew.parent, xNew, self.xGoal, self.distGoalToApp):
                    
                    self.treeVertexGoal.append(xNew)
                    xNearestPrime = self.nearest_node(self.treeVertexStart, xNew)
                    xNewPrime = self.steer(xNearestPrime, xNew)
                    xNewPrime.parent = xNearestPrime

                    if not self.is_config_in_collision(xNewPrime) and \
                        not self.is_connect_config_in_collision(xNewPrime.parent, xNewPrime) and \
                        not self.is_config_in_region_of_config(xNewPrime, self.xGoal, self.distGoalToApp) and \
                        not self.is_connect_config_in_region_of_config(xNewPrime.parent, xNewPrime, self.xGoal, self.distGoalToApp):                   
                        
                        self.treeVertexStart.append(xNewPrime)
                        while True:
                            xNewPPrime = self.steer(xNewPrime, xNew)
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

    def optimizer_greedy_prune_path(self, initialPath): # lost a lot of information about the collision when curve fit which as expected
        prunedPath = [initialPath[0]]
        indexNext = 1

        while indexNext != len(initialPath):
            if self.is_connect_config_in_collision(prunedPath[-1], initialPath[indexNext], NumSeg=int(self.distance_between_config(prunedPath[-1], initialPath[indexNext])/self.eta)):
                prunedPath.append(initialPath[indexNext-1])
            else:
                indexNext += 1

        prunedPath.extend([initialPath[-2], initialPath[-1]]) # add back xApp and xGoal to path from the back

        return prunedPath

    def optimizer_mid_node_prune_path(self, path): # remove an middle node (even index)
        prunedPath = path.copy() # create copy of list, not optimized but just to save the original for later used maybe
        for i in range(len(prunedPath) - 3, -1, -2):
            prunedPath.pop(i)

        return prunedPath

    def optimizer_segment_linear_interpolation_path(self, path, numSeg=10): # linear interpolate between each node in path for number of segment
        segmentedPath = []
        currentIndex = 0
        nextIndex = 1
        while nextIndex != len(path):
            segmentedPath.append(path[currentIndex])
            distX, distY, distZ, distP, distQ, distR = self.distance_each_component_between_config(path[currentIndex], path[nextIndex])
            rateX = distX / numSeg
            rateY = distY / numSeg
            rateZ = distZ / numSeg
            rateP = distP / numSeg
            rateQ = distQ / numSeg
            rateR = distR / numSeg
            for i in range(1, numSeg - 1):
                newX = path[currentIndex].x + (rateX*i)
                newY = path[currentIndex].y + (rateY*i)
                newZ = path[currentIndex].z + (rateZ*i)
                newP = path[currentIndex].p + (rateP*i)
                newQ = path[currentIndex].q + (rateQ*i)
                newR = path[currentIndex].r + (rateR*i)
                xNew = Node(newX, newY, newZ, newP, newQ, newR)
                segmentedPath.append(xNew)
            currentIndex += 1
            nextIndex += 1

        return segmentedPath

    # def optimizer_reject_unoptimal_prune_path(self, path):

    #     return prunedPath
    
    # def optimizer_apply_approach_path(self, path): # assume the pass in path is [xinit, x1, x2, ..., xgoal]
    #     pathToApproach = path.copy()
    #     # remove the last element since we know it is the xgoal
    #     pathToApproach.pop()
    #     radiusToApproach = self.distance_between_config(self.xGoal, self.xApp)
    #     for i in range(-1, -len(pathToApproach), -1):
    #         steerToCandidate = pathToApproach[i]
    #         pathToApproach.remove(steerToCandidate)
    #         if self.distance_between_config(self.xGoal, steerToCandidate) >= radiusToApproach:
    #             break

    #     xToMerge = self.steer_to_exact_distance(self.xGoal, steerToCandidate, radiusToApproach)

    #     # we need the method to move from xToMerge to xApp, we can not move directly.
    #     xIntermidiary = move(xToMerge, self.xApp)
    #     pathToApproach = pathToApproach + [xToMerge] + [xIntermidiary] + [self.xApp, self.xGoal]

    #     return pathToApproach
    
    def hypersphere_surface_sampling(self, centerNode, radius):
        randPoint = np.random.normal(size=6)
        unitRandPoint = randPoint / np.linalg.norm(randPoint)
        point = (unitRandPoint * radius) + np.array([centerNode.x, centerNode.y, centerNode.z, centerNode.p, centerNode.q, centerNode.r])
        xNew = Node(point[0], point[1], point[2], point[3], point[4], point[5])
        return xNew
    
    def uni_sampling(self):
        x = np.random.uniform(low=self.xMinRange, high=self.xMaxRange)
        y = np.random.uniform(low=self.yMinRange, high=self.yMaxRange)
        z = np.random.uniform(low=self.zMinRange, high=self.zMaxRange)
        p = np.random.uniform(low=self.pMinRange, high=self.pMaxRange)
        q = np.random.uniform(low=self.qMinRange, high=self.qMaxRange)
        r = np.random.uniform(low=self.rMinRange, high=self.rMaxRange)

        xRand = Node(x, y, z, p, q, r)
        return xRand

    def bias_sampling(self, biasTowardNode):
        if np.random.uniform(low=0, high=1.0) < self.probabilityGoalBias:
            xRand = biasTowardNode
        else:
            xRand = self.uni_sampling()
        return xRand

    def tree_swap_flag(self):
        if self.treeSwapFlag is True:
            self.treeSwapFlag = False
        elif self.treeSwapFlag is False:
            self.treeSwapFlag = True

    def nearest_node(self, treeVertex, xRand):
        vertexList = []

        for eachVertex in treeVertex:
            vertexList.append(self.distance_between_config(xRand, eachVertex))

        minIndex = np.argmin(vertexList)
        xNear = treeVertex[minIndex]

        return xNear

    def steer(self, xNearest, xRand):
        distX, distY, distZ, distP, distQ, distR = self.distance_each_component_between_config(xNearest, xRand)
        dist = np.linalg.norm([distX, distY, distZ, distP, distQ, distR])
        if dist <= self.eta:
            # xNew = xRand
            xNew = Node(xRand.x, xRand.y, xRand.z, xRand.p, xRand.q, xRand.r)
        else:
            dX = (distX/dist) * self.eta
            dY = (distY/dist) * self.eta
            dZ = (distZ/dist) * self.eta
            dP = (distP/dist) * self.eta
            dQ = (distQ/dist) * self.eta
            dR = (distR/dist) * self.eta
            newX = xNearest.x + dX
            newY = xNearest.y + dY
            newZ = xNearest.z + dZ
            newP = xNearest.p + dP
            newQ = xNearest.q + dQ
            newR = xNearest.r + dR
            xNew = Node(newX, newY, newZ, newP, newQ, newR)

        return xNew
    
    def steer_to_exact_distance(self, xStart, xGoal, distance):
        distX, distY, distZ, distP, distQ, distR = self.distance_each_component_between_config(xStart, xGoal)
        dist = np.linalg.norm([distX, distY, distZ, distP, distQ, distR])
        dX = (distX/dist) * distance
        dY = (distY/dist) * distance
        dZ = (distZ/dist) * distance
        dP = (distP/dist) * distance
        dQ = (distQ/dist) * distance
        dR = (distR/dist) * distance
        newX = xStart.x + dX
        newY = xStart.y + dY
        newZ = xStart.z + dZ
        newP = xStart.p + dP
        newQ = xStart.q + dQ
        newR = xStart.r + dR
        xNew = Node(newX, newY, newZ, newP, newQ, newR)

        return xNew

    def is_both_tree_node_near(self, return_near_node=False):
        for eachVertexStart in self.treeVertexStart:
            for eachVertexGoal in self.treeVertexGoal:
                if self.distance_between_config(eachVertexStart, eachVertexGoal) <= self.eta:
                    if return_near_node:
                        return eachVertexStart, eachVertexGoal
                    return True
        return False

    def is_config_in_region_of_config(self, xToCheck, xCenter, radius=None):
        if radius is None:
            radius = self.eta
        if self.distance_between_config(xToCheck, xCenter) < radius:
            return True
        return False

    def is_connect_config_in_region_of_config(self, xToCheckStart, xToCheckEnd, xCenter, radius=None, NumSeg=10):
        if radius is None:
            radius = self.eta
        distX, distY, distZ, distP, distQ, distR = self.distance_each_component_between_config(xToCheckStart, xToCheckEnd)
        rateX = distX / NumSeg
        rateY = distY / NumSeg
        rateZ = distZ / NumSeg
        rateP = distP / NumSeg
        rateQ = distQ / NumSeg
        rateR = distR / NumSeg
        for i in range(1, NumSeg - 1):
            newX = xToCheckStart.x + (rateX*i)
            newY = xToCheckStart.y + (rateY*i)
            newZ = xToCheckStart.z + (rateZ*i)
            newP = xToCheckStart.p + (rateP*i)
            newQ = xToCheckStart.q + (rateQ*i)
            newR = xToCheckStart.r + (rateR*i)
            xNew = Node(newX, newY, newZ, newP, newQ, newR)
            if self.is_config_in_region_of_config(xNew, xCenter, radius):
                return True
        return False

    def is_config_in_collision(self, xNew):
        # check if collision status is available to use without calling the oracle
        # for i, config in enumerate(self.configSearched):
        #     if self.distance_between_config(xNew, config) <= self.eta:
        #         return self.collisionState[i]

        # if there are not collision information, then call the oracle to check
        timeStartKCD = time.perf_counter_ns()
        theta = np.array([xNew.x, xNew.y, xNew.z, xNew.p, xNew.q, xNew.r]).reshape(6, 1)
        linkPose = self.robot.forward_kinematic(theta, return_link_pos=True)
        linearm1 = ObjLine2D(linkPose[0][0], linkPose[0][1], linkPose[1][0], linkPose[1][1])
        linearm2 = ObjLine2D(linkPose[1][0], linkPose[1][1], linkPose[2][0], linkPose[2][1])
        linearm3 = ObjLine2D(linkPose[2][0], linkPose[2][1], linkPose[3][0], linkPose[3][1])
        linearm4 = ObjLine2D(linkPose[3][0], linkPose[3][1], linkPose[4][0], linkPose[4][1])
        linearm5 = ObjLine2D(linkPose[4][0], linkPose[4][1], linkPose[5][0], linkPose[5][1])
        linearm6 = ObjLine2D(linkPose[5][0], linkPose[5][1], linkPose[6][0], linkPose[6][1])

        link = [linearm1, linearm2, linearm3, linearm4, linearm5, linearm6]

        # add collsion state to database
        self.configSearched.append(xNew)

        # obstacle collision
        for obs in self.taskMapObs:
            for i in range(len(link)):
                if intersect_line_v_rectangle(link[i], obs):
                    self.collisionState.append(True)
                    return True

        timeEndKCD = time.perf_counter_ns()
        self.perfMatrix["KCDTimeSpend"] += timeEndKCD - timeStartKCD

        self.collisionState.append(False)

        return False

    def is_connect_config_in_collision(self, xNearest, xNew, NumSeg=10):
        distX, distY, distZ, distP, distQ, distR = self.distance_each_component_between_config(xNearest, xNew)
        rateX = distX / NumSeg
        rateY = distY / NumSeg
        rateZ = distZ / NumSeg
        rateP = distP / NumSeg
        rateQ = distQ / NumSeg
        rateR = distR / NumSeg
        for i in range(1, NumSeg - 1):
            newX = xNearest.x + (rateX*i)
            newY = xNearest.y + (rateY*i)
            newZ = xNearest.z + (rateZ*i)
            newP = xNearest.p + (rateP*i)
            newQ = xNearest.q + (rateQ*i)
            newR = xNearest.r + (rateR*i)
            xNew = Node(newX, newY, newZ, newP, newQ, newR)
            if self.is_config_in_collision(xNew):
                return True
        return False

    def distance_between_config(self, xStart, xEnd):
        return np.linalg.norm([xStart.x - xEnd.x,
                               xStart.y - xEnd.y,
                               xStart.z - xEnd.z,
                               xStart.p - xEnd.p,
                               xStart.q - xEnd.q,
                               xStart.r - xEnd.r])

    def distance_each_component_between_config(self, xStart, xEnd):
        return xEnd.x - xStart.x, \
               xEnd.y - xStart.y, \
               xEnd.z - xStart.z, \
               xEnd.p - xStart.p, \
               xEnd.q - xStart.q, \
               xEnd.r - xStart.r

if __name__ == "__main__":
    np.random.seed(9)
    from robot.planar_sixdof import PlanarSixDof
    from planner.extract_path_class import extract_path_class_6d
    import matplotlib.pyplot as plt
    from map.taskmap_geo_format import two_near_ee_for_devplanner
    from planner_util.plot_util import plot_joint_6d
    from scipy.optimize import curve_fit
    from util.dictionary_pretty import print_dict

    robot = PlanarSixDof()

    thetaInit = np.array([0, 0, 0, 0, 0, 0]).reshape(6, 1)
    thetaGoal = np.array([1.2, 0, -0.2, 0, -1.2, -0.2]).reshape(6, 1)
    thetaApp = np.array([1.3, 0, -0.3, 0, -1.21, -0.16]).reshape(6, 1)
    obsList = two_near_ee_for_devplanner()

    # plot pre planning
    fig1, ax1 = plt.subplots()
    ax1.set_aspect("equal")
    ax1.set_title("Pre Planning Plot")
    robot.plot_arm(thetaInit, plt_axis=ax1)
    robot.plot_arm(thetaApp, plt_axis=ax1)
    robot.plot_arm(thetaGoal, plt_axis=ax1)
    for obs in obsList:
        obs.plot()
    plt.show()

    planner = DevPlanner(robot, obsList, thetaInit, thetaApp, thetaGoal, eta=0.1, maxIteration=5000)
    path = planner.planning()
    print(f"==>> path: \n{path}")
    print_dict(planner.perfMatrix)

    pathX, pathY, pathZ, pathP, pathQ, pathR = extract_path_class_6d(path)

    # plot after planning
    fig2, ax2 = plt.subplots()
    ax2.set_aspect("equal")
    ax2.set_title("After Planning Plot")
    for obs in obsList:
        obs.plot()
    robot.plot_arm(thetaInit, plt_axis=ax2)
    for i in range(len(path)):
        robot.plot_arm(np.array([pathX[i], pathY[i], pathZ[i], pathP[i], pathQ[i], pathR[i]]).reshape(6, 1), plt_axis=ax2)
        plt.pause(0.5)
    plt.show()

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
    poptX, pcovX = curve_fit(quintic5deg, time, pathX, sigma=sigma)
    poptY, pcovY = curve_fit(quintic5deg, time, pathY, sigma=sigma)
    poptZ, pcovZ = curve_fit(quintic5deg, time, pathZ, sigma=sigma)
    poptP, pcovP = curve_fit(quintic5deg, time, pathP, sigma=sigma)
    poptQ, pcovQ = curve_fit(quintic5deg, time, pathQ, sigma=sigma)
    poptR, pcovR = curve_fit(quintic5deg, time, pathR, sigma=sigma)

    timeSmooth = np.linspace(0, 1, 100)
    # plot after planning
    fig3, ax3 = plt.subplots()
    ax3.set_aspect("equal")
    ax3.set_title("After Planning Plot")
    for obs in obsList:
        obs.plot()
    for i in range(timeSmooth.shape[0]):
        robot.plot_arm(np.array([quintic5deg(timeSmooth[i], *poptX),
                                 quintic5deg(timeSmooth[i], *poptY),
                                 quintic5deg(timeSmooth[i], *poptZ),
                                 quintic5deg(timeSmooth[i], *poptP),
                                 quintic5deg(timeSmooth[i], *poptQ),
                                 quintic5deg(timeSmooth[i], *poptR)]).reshape(6, 1), plt_axis=ax3)
        plt.pause(0.1)
    plt.show()

    fig4, axes = plt.subplots(6, 1, sharex='all')
    axes[0].plot(time, pathX, 'ro')
    axes[0].plot(time, quintic5deg(time, *poptX))
    axes[1].plot(time, pathY, 'ro')
    axes[1].plot(time, quintic5deg(time, *poptY))
    axes[2].plot(time, pathZ, 'ro')
    axes[2].plot(time, quintic5deg(time, *poptZ))
    axes[3].plot(time, pathP, 'ro')
    axes[3].plot(time, quintic5deg(time, *poptP))
    axes[4].plot(time, pathQ, 'ro')
    axes[4].plot(time, quintic5deg(time, *poptQ))
    axes[5].plot(time, pathR, 'ro')
    axes[5].plot(time, quintic5deg(time, *poptR))
    plt.show()