import os
import sys

wd = os.path.abspath(os.getcwd())
sys.path.append(str(wd))

import time
import numpy as np
import matplotlib.patches as mpatches
from copsim.arm_api import UR5eArmCoppeliaSimAPI
from planner.planner2d.arm_env2d import RobotArm2DEnvironment, TaskSpace2DEnvironment


class Node:

    def __init__(self, config, parent=None, cost=0.0) -> None:
        self.config = config
        self.parent = parent
        self.child = []
        self.cost = cost

    def __repr__(self) -> str:
        return f'\nconfig = {self.config.T}, hasParent = {True if self.parent != None else False}, NumChild = {len(self.child)}'


class RRTComponent:

    def __init__(self, eta, subEta, maxIteration, numDoF, envChoice, nearGoalRadius, rewireRadius, endIterationID, print_debug) -> None:
        # Check DOF [min, max]
        if numDoF == 2:
            self.jointRange = [[-np.pi, np.pi], [-np.pi, np.pi]]
        elif numDoF == 3:
            self.jointRange = [[-np.pi, np.pi], [-np.pi, np.pi], [-np.pi, np.pi]]
        elif numDoF == 6:
            self.jointRange = [[-np.pi, np.pi], [-np.pi, np.pi], [-np.pi, np.pi], [-np.pi, np.pi], [-np.pi, np.pi], [-np.pi, np.pi]]

        # Check Environment CopSim or Plannar
        if envChoice == "CopSim":
            self.robotEnv = UR5eArmCoppeliaSimAPI()
        elif envChoice == "Planar":
            # self.robotEnv = RobotArm2DEnvironment()
            self.robotEnv = TaskSpace2DEnvironment()

        # planner properties
        self.eta = eta
        self.subEta = subEta
        self.maxIteration = maxIteration
        self.nearGoalRadius = nearGoalRadius
        self.rewireRadius = rewireRadius
        self.probabilityGoalBias = 0.05  # When close to goal select goal, with this probability? default: 0.05, must modify my code
        self.maxGoalSample = 10  #Goal samples are only sampled until maxSampleCount() goals are in the tree, to prohibit duplicate goal states.
        self.endIterationID = endIterationID # break condition
        self.terminateNumSolutions = 5

        # collision database
        self.configSearched = []
        self.collisionState = []

        # performance matrix
        self.perfMatrix = {
            "Planner Name": "",
            "Parameters": {
                "eta": self.eta,
                "subEta": self.subEta,
                "Max Iteration": self.maxIteration,
                "Rewire Radius": self.rewireRadius
            },
            "Number of Node": 0,
            "Total Planning Time": 0.0,  # include KCD
            "KCD Time Spend": 0.0,
            "Planning Time Only": 0.0,
            "Number of Collision Check": 0,
            "Average KCD Time": 0.0,
            "Cost Graph": []
        }

        # keep track cost
        self.cBestPrevious = np.inf
        self.cBestNow = np.inf
        self.costDiffConstant = None # if cost different is lower than this constant, terminate loop

        # misc, for improve unnessary computation
        self.DoF = numDoF
        self.jointIndex = range(self.DoF)
        self.gammaFunction = {1: 1, 2: 1, 2.5:1.32934, 3: 2, 4: 6, 5: 24, 6: 120}  # given DoF, apply gamma(DoF)

        # debug setting
        self.print_debug = print_debug

    def uni_sampling(self) -> Node:
        config = np.array([[np.random.uniform(low=j[0], high=j[1])] for j in self.jointRange])
        xRand = Node(config)
        return xRand

    def bias_uniform_sampling(self, biasTowardNode, numNodeInGoalRegion):
        if numNodeInGoalRegion < self.maxGoalSample and np.random.uniform(low=0, high=1.0) < self.probabilityGoalBias:
            xRand = Node(biasTowardNode.config)
        else:
            xRand = self.uni_sampling()
        return xRand

    def informed_sampling(self, xCenter, cMax, cMin, rotationAxisC):
        L = self.hyperellipsoid_axis_length(cMax, cMin)
        while True:
            xBall = self.unit_ball_sampling()
            xRand = (rotationAxisC@L@xBall) + xCenter
            xRand = Node(xRand)
            if self.is_config_in_joint_limit(xRand):
                break
        return xRand

    def bias_informed_sampling(self, xCenter, cMax, cMin, rotationAxisC, biasTowardNode, numNodeInGoalRegion):
        if numNodeInGoalRegion < self.maxGoalSample and np.random.uniform(low=0, high=1.0) < self.probabilityGoalBias:
            xRand = Node(biasTowardNode.config)
        else:
            xRand = self.informed_sampling(xCenter, cMax, cMin, rotationAxisC)
        return xRand
    
    def unit_ball_sampling(self):
        u = np.random.normal(0.0, 1.0, (self.DoF + 2, 1))
        norm = np.linalg.norm(u)
        u = u / norm
        return u[:self.DoF,:] #The first N coordinates are uniform in a unit N ball

    def rotation_to_world(self, xStart, xGoal): # C
        cMin = self.distance_between_config(xStart, xGoal)
        a1 = (xGoal.config - xStart.config) / cMin
        I1 = np.array([1.0] + [0.0] * (self.DoF - 1)).reshape(1, -1)
        M = a1 @ I1
        U, _, V_T = np.linalg.svd(M, True, True)
        middleTerm = [1.0] * (self.DoF - 1) + [np.linalg.det(U) * np.linalg.det(V_T.T)]
        return U @ np.diag(middleTerm) @ V_T  

    def hyperellipsoid_axis_length(self, cMax, cMin):  # L
        r1 = cMax / 2
        ri = np.sqrt(cMax**2 - cMin**2) / 2
        diagTerm = [r1] + [ri] * (self.DoF - 1)
        return np.diag(diagTerm)

    def local_path_sampling(self, anchorPath, localPath, NumSeg):  #expected given path [xinit, x1, x2, ..., xcandidateToxApp, xApp]
        gRand = np.random.randint(low=0, high=NumSeg)
        randDownPercentage = np.random.uniform(low=0.0, high=1.0)
        randAlongPercentage = np.random.uniform(low=0.0, high=1.0)

        if gRand == 0:  # start Gap
            downRandRight = self.steer_node_in_between_percentage(localPath[1], anchorPath[1], randDownPercentage)
            xRand = self.steer_node_in_between_percentage(localPath[0], downRandRight, randAlongPercentage)
        elif gRand == NumSeg - 1:  # end Gap
            downRandLeft = self.steer_node_in_between_percentage(localPath[-2], anchorPath[-2], randDownPercentage)
            xRand = self.steer_node_in_between_percentage(downRandLeft, localPath[-1], randAlongPercentage)
        else:  # mid Gap
            downRandLeft = self.steer_node_in_between_percentage(localPath[gRand], anchorPath[gRand], randDownPercentage)
            downRandRight = self.steer_node_in_between_percentage(localPath[gRand + 1], anchorPath[gRand + 1], randDownPercentage)
            xRand = self.steer_node_in_between_percentage(downRandLeft, downRandRight, randAlongPercentage)

        return xRand

    def unit_nball_volume_measure(self): # The Lebesgue measure (i.e., "volume") of an n-dimensional ball with a unit radius.
        return (np.pi**(self.DoF / 2)) / self.gammaFunction[(self.DoF / 2) + 1]  # ziD

    def prolate_hyperspheroid_measure(self): # The Lebesgue measure (i.e., "volume") of an n-dimensional prolate hyperspheroid (a symmetric hyperellipse) given as the distance between the foci and the transverse diameter.
        pass

    def lebesgue_obstacle_free_measure(self):
        diff = np.diff(self.jointRange)
        return np.prod(diff)

    def calculate_rewire_radius(self, numVertex, rewireFactor=1.1):
        inverseDoF = 1.0 / self.DoF
        gammaRRG = rewireFactor * 2.0 * ((1.0+inverseDoF) * (self.lebesgue_obstacle_free_measure() / self.unit_nball_volume_measure()))**(inverseDoF)
        return np.min([self.eta, gammaRRG * (np.log(numVertex) / numVertex)**(inverseDoF)])

    def termination_check(self, solutionList):
        if self.endIterationID == 1: # maxIteration exceeded
            return self.termination_on_max_iteration()
        elif self.endIterationID == 2: # first solution found
            return self.termination_on_first_solution(solutionList)
        elif self.endIterationID == 3: #cost drop different is low
            return self.termination_on_cost_drop_different()

    def termination_on_max_iteration(self): # already in for loop, no break is needed
        return False 
    
    def termination_on_first_solution(self, solutionList):
        if len(solutionList) == self.terminateNumSolutions:
            return True
        else:
            return False

    def termination_on_cost_drop_different(self):
        if self.cBestPrevious - self.cBestNow < self.costDiffConstant:
            return True
        else:
            return False

    def nearest_node(self, treeVertices, xCheck, returnDistList=False):
        distListToxCheck = [self.distance_between_config(xCheck, x) for x in treeVertices]
        minIndex = np.argmin(distListToxCheck)
        xNearest = treeVertices[minIndex]
        if returnDistList:
            return xNearest, distListToxCheck
        else:
            return xNearest

    def steer_node_in_between_percentage(self, xFrom, xTo, percentage):  # percentage only between [0.0 - 1.0]
        distI = self.distance_each_component_between_config(xFrom, xTo)
        newI = xFrom.config + percentage*distI
        xNew = Node(newI)
        return xNew

    def steer(self, xFrom, xTo, distance, returnxNewIsxRand=False):
        distI = self.distance_each_component_between_config(xFrom, xTo)
        dist = np.linalg.norm(distI)
        if dist <= distance:
            xNew = Node(xTo.config)
        else:
            dI = (distI/dist) * distance
            newI = xFrom.config + dI
            xNew = Node(newI)
        if returnxNewIsxRand:
            return xNew, False
        else:
            return xNew

    def near(self, treeVertices, xCheck, searchRadius=None, numTopNearest=10, distListToxCheck=None, dynamicSubset=False):
        if searchRadius is None:
            searchRadius = self.calculate_rewire_radius(len(treeVertices))

        if distListToxCheck:
            distListToxCheck = np.array(distListToxCheck)
        else:
            distListToxCheck = np.array([self.distance_between_config(xCheck, vertex) for vertex in treeVertices])

        nearIndices = np.where(distListToxCheck<=searchRadius)[0]

        if dynamicSubset:
            return [treeVertices[item] for item in nearIndices]
        else:
            if len(nearIndices) < numTopNearest:
                return [treeVertices[item] for item in nearIndices]
            else:
                sortedDist = np.argsort(distListToxCheck)[:numTopNearest]
                return [treeVertices[item] for item in sortedDist]

    def cost_line(self, xFrom, xTo):
        return self.distance_between_config(xFrom, xTo)

    def is_collision_and_in_goal_region(self, xFrom, xTo, xCenter, radius):  # return True if any of it is true
        if self.is_config_in_region_of_config(xTo, xCenter, radius):
            return True
        elif self.is_connect_config_in_region_of_config(xFrom, xTo, xCenter, radius):
            return True
        elif self.is_config_in_collision(xTo):
            return True
        elif self.is_connect_config_in_collision(xFrom, xTo):
            return True
        else:
            return False

    def is_collision(self, xFrom, xTo):
        if self.is_config_in_collision(xTo):
            return True
        elif self.is_connect_config_in_collision(xFrom, xTo):
            return True
        else:
            return False

    def is_config_in_region_of_config(self, xCheck, xCenter, radius):
        if self.distance_between_config(xCheck, xCenter) < radius:
            return True
        return False

    def is_connect_config_in_region_of_config(self, xCheckFrom, xCheckTo, xCenter, radius, NumSeg=None):
        limitedNumSeg = 10
        distI = self.distance_each_component_between_config(xCheckFrom, xCheckTo)
        dist = np.linalg.norm(distI)
        if NumSeg:
            NumSeg = NumSeg
        else:
            NumSeg = int(np.ceil(dist / self.subEta))
            if NumSeg > limitedNumSeg:
                NumSeg = limitedNumSeg
        rateI = distI / NumSeg
        for i in range(1, NumSeg):
            newI = xCheckFrom.config + rateI*i
            xNew = Node(newI)
            if self.is_config_in_region_of_config(xNew, xCenter, radius):
                return True
        return False

    def is_config_in_collision(self, xCheck):
        timeStartKCD = time.perf_counter_ns()
        result = self.robotEnv.collision_check(xCheck)
        timeEndKCD = time.perf_counter_ns()
        self.perfMatrix["KCD Time Spend"] += timeEndKCD - timeStartKCD
        self.perfMatrix["Number of Collision Check"] += 1
        return result

    def is_connect_config_in_collision(self, xFrom, xTo, NumSeg=None):
        limitedNumSeg = 10
        distI = self.distance_each_component_between_config(xFrom, xTo)
        dist = np.linalg.norm(distI)
        if NumSeg:
            NumSeg = NumSeg
        else:
            NumSeg = int(np.ceil(dist / self.subEta))
            if NumSeg > limitedNumSeg:
                NumSeg = limitedNumSeg
        rateI = distI / NumSeg
        for i in range(1, NumSeg):
            newI = xFrom.config + rateI*i
            xTo = Node(newI)
            if self.is_config_in_collision(xTo):
                return True
        return False

    def distance_between_config(self, xFrom, xTo):
        return np.linalg.norm(self.distance_each_component_between_config(xFrom, xTo))

    def distance_each_component_between_config(self, xFrom, xTo):
        return xTo.config - xFrom.config

    def star_optimizer(self, treeToAdd, xNew, rewireRadius, xNewIsxRand=None, preCalDistListToxNew=None):
        # optimized to reduce dist cal if xNew is xRand
        if xNewIsxRand:
            XNear = self.near(treeToAdd, xNew, rewireRadius, preCalDistListToxNew)
        else:
            XNear = self.near(treeToAdd, xNew, rewireRadius)

        # Parenting
        cMin = xNew.cost
        XNearCostToxNew = [xNear.cost + self.cost_line(xNear, xNew) for xNear in XNear]
        XNearCollisionState = [None] * len(XNear)
        XNearCostToxNewSortedIndex = np.argsort(XNearCostToxNew)
        for index in XNearCostToxNewSortedIndex:
            if XNearCostToxNew[index] < cMin:
                collisionState = self.is_connect_config_in_collision(XNear[index], xNew)
                XNearCollisionState[index] = collisionState
                if not collisionState:
                    xNew.parent.child.remove(xNew)
                    xNew.parent = XNear[index]
                    xNew.cost = XNearCostToxNew[index]
                    XNear[index].child.append(xNew)
                    break

        treeToAdd.append(xNew)
        
        # Rewiring
        for index, xNear in enumerate(XNear):
            cNew = xNew.cost + self.cost_line(xNew, xNear)
            if cNew < xNear.cost:
                if XNearCollisionState[index] is None:
                    collisionState = self.is_connect_config_in_collision(xNear, xNew)
                else:
                    collisionState = XNearCollisionState[index]
                if collisionState is False:
                    xNear.parent.child.remove(xNear)
                    xNear.parent = xNew
                    xNear.cost = cNew
                    xNew.child.append(xNear)
                    self.update_child_cost(xNear)

    def tree_swap_flag(self):
        if self.treeSwapFlag is True:
            self.treeSwapFlag = False
        elif self.treeSwapFlag is False:
            self.treeSwapFlag = True

    def is_config_in_joint_limit(self, xCheck):
        for j in self.jointIndex:
            if not self.jointRange[j][0] < xCheck.config[j] < self.jointRange[j][1]:
                return False
        return True

    def reparent_merge_tree(self, xTobeParent, xNow, treeToAddTo):
        while True:
            if xNow.parent is None:
                xNow.parent = xTobeParent
                xNow.cost = xNow.parent.cost + self.cost_line(xNow.parent, xNow)
                xTobeParent.child.append(xNow)
                self.update_child_cost(xNow, treeToAddTo)
                # xParentSave.child.remove(xNow) # Xnow is Xapp which has no parent so we dont have to do this
                treeToAddTo.append(xNow)
                break

            xParentSave = xNow.parent
            xNow.parent = xTobeParent
            xNow.cost = xNow.parent.cost + self.cost_line(xNow.parent, xNow)
            xTobeParent.child.append(xNow)
            self.update_child_cost(xNow, treeToAddTo)
            xParentSave.child.remove(xNow)
            treeToAddTo.append(xNow)

            # update for next iteration
            xTobeParent = xNow
            xNow = xParentSave

    def update_child_cost(self, xCheck, treeToAdd=None): # recursively updates the cost of the children of this node if the cost up to this node has changed.
        for child in xCheck.child:
            child.cost = child.parent.cost + self.cost_line(child.parent, child)
            if treeToAdd:
                treeToAdd.append(child)
            self.update_child_cost(child)

    def search_backtrack_single_directional_path(self, backFromNode, attachNode=None):  # return path is [xinit, x1, x2, ..., xapp, xgoal]
        pathStart = [backFromNode]
        currentNodeStart = backFromNode
        while currentNodeStart.parent is not None:
            currentNodeStart = currentNodeStart.parent
            pathStart.append(currentNodeStart)

        pathStart.reverse()

        if attachNode:
            return pathStart + [attachNode]  # attachNode is Normally xGoal
        else:
            return pathStart

    def search_backtrack_bidirectional_path(self, backFromNodeTa, backFromNodeTb, attachNode=None):  # return path is [xinit, x1, x2, ..., xapp, xgoal]
        # backFromNodeTa = self.connectNodeStart # nearStart
        # backFromNodeTb = self.connectNodeGoal  # nearGoal

        pathStart = [backFromNodeTa]
        currentNodeStart = backFromNodeTa
        while currentNodeStart.parent is not None:
            currentNodeStart = currentNodeStart.parent
            pathStart.append(currentNodeStart)

        pathGoal = [backFromNodeTb]
        currentNodeGoal = backFromNodeTb
        while currentNodeGoal.parent is not None:
            currentNodeGoal = currentNodeGoal.parent
            pathGoal.append(currentNodeGoal)

        pathStart.reverse()

        if attachNode:
            return pathStart + pathGoal + [attachNode]  # attachNode is Normally xGoal
        else:
            return pathStart + pathGoal

    def search_best_cost_singledirection_path(self, backFromNode, treeVertexList, attachNode=None):
        vertexListCost = [vertex.cost + self.cost_line(vertex, backFromNode) for vertex in treeVertexList]
        # costSortedIndex = sorted(range(len(vertexListCost)), key=lambda i: vertexListCost[i]) the same as argsort
        costSortedIndex = np.argsort(vertexListCost)

        for xNearIndex in costSortedIndex:
            if not self.is_connect_config_in_collision(treeVertexList[xNearIndex], backFromNode):
                path = [backFromNode, treeVertexList[xNearIndex]]
                currentNode = treeVertexList[xNearIndex]
                while currentNode.parent is not None:
                    currentNode = currentNode.parent
                    path.append(currentNode)
                path.reverse()

                if attachNode:
                    return path + [attachNode]  # attachNode is Normally xGoal
                else:
                    return path

    def search_best_cost_bidirection_path(self, connectNodePairList, attachNode=None):
        vertexPairListCost = [vertexA.cost + vertexB.cost + self.cost_line(vertexA, vertexB) for vertexA, vertexB in connectNodePairList]
        costMinIndex = np.argmin(vertexPairListCost)

        pathStart = [connectNodePairList[costMinIndex][0]]
        currentNodeStart = connectNodePairList[costMinIndex][0]
        while currentNodeStart.parent is not None:
            currentNodeStart = currentNodeStart.parent
            pathStart.append(currentNodeStart)

        pathGoal = [connectNodePairList[costMinIndex][1]]
        currentNodeGoal = connectNodePairList[costMinIndex][1]
        while currentNodeGoal.parent is not None:
            currentNodeGoal = currentNodeGoal.parent
            pathGoal.append(currentNodeGoal)

        pathStart.reverse()

        if attachNode:
            return pathStart + pathGoal + [attachNode]  # attachNode is Normally xGoal
        else:
            return pathStart + pathGoal

    def segment_interpolation_between_config(self, xStart, xEnd, NumSeg, includexStart=False):  # calculate line interpolation between two config
        if includexStart:
            anchorPath = [xStart]
        else:  # only give [x1, x2,..., xEnd] - xStart is excluded
            anchorPath = []

        distI = self.distance_each_component_between_config(xStart, xEnd)
        rateI = distI / NumSeg
        for i in range(1, NumSeg):
            newI = xStart.config + rateI*i
            xMid = Node(newI)
            anchorPath.append(xMid)
        anchorPath.append(xEnd)

        return anchorPath

    def segment_interpolation_path(self, path, numSeg=10):  # linear interpolate between each node in path for number of segment
        segmentedPath = []
        currentIndex = 0
        nextIndex = 1
        while nextIndex != len(path):
            segmentedPath.append(path[currentIndex])
            distI = self.distance_each_component_between_config(path[currentIndex], path[nextIndex])
            rateI = distI / numSeg
            for i in range(1, numSeg):
                newI = path[currentIndex].config + (rateI*i)
                xNew = Node(newI)
                segmentedPath.append(xNew)
            currentIndex += 1
            nextIndex += 1

        return segmentedPath
    
    def postprocess_greedy_prune_path(self, initialPath):  # lost a lot of information about the collision when curve fit which as expected
        prunedPath = [initialPath[0]]
        indexNext = 1
        while indexNext != len(initialPath):
            if self.is_connect_config_in_collision(prunedPath[-1], initialPath[indexNext], NumSeg=int(self.distance_between_config(prunedPath[-1], initialPath[indexNext]) / self.eta)):
                prunedPath.append(initialPath[indexNext - 1])
            else:
                indexNext += 1
        prunedPath.extend([initialPath[-2], initialPath[-1]])  # add back xApp and xGoal to path from the back
        return prunedPath

    def postprocess_mid_node_prune_path(self, path):  # remove an middle node (even index)
        prunedPath = path.copy()  # create copy of list, not optimized but just to save the original for later used maybe
        for i in range(len(prunedPath) - 3, -1, -2):
            prunedPath.pop(i)
        return prunedPath

    def cbest_single_tree(self, treeVertices, nodeToward, iteration, print_debug=False): # search in treeGoalRegion for the current best cost
        if len(treeVertices) == 0:
            cBest = np.inf
            xSolnCost = []
        else:
            xSolnCost = [xSoln.cost + self.cost_line(xSoln, nodeToward) for xSoln in treeVertices]
            cBest = min(xSolnCost)
            if cBest < self.cBestPrevious : # this has nothing to do with planning itself, just for record performance data only
                self.perfMatrix["Cost Graph"].append((iteration, cBest))
                self.cBestPrevious = cBest
        if print_debug:
            print(f"Iteration : [{iteration}] - Best Cost : [{cBest}]", end='\r', flush=True)
        return cBest

    def cbest_dual_tree(self, connectNodePair, iteration, print_debug=False): # search in connectNodePairList for the current best cost
        if len(connectNodePair) == 0:
            cBest = np.inf
            xSolnCost = []
        else:
            xSolnCost = [vertexA.cost + vertexB.cost + self.cost_line(vertexA, vertexB) for vertexA, vertexB in connectNodePair]
            cBest = min(xSolnCost)
            if cBest < self.cBestPrevious:
                self.perfMatrix["Cost Graph"].append((iteration, cBest))
                self.cBestPrevious = cBest
        if print_debug:
            print(f"Iteration : [{iteration}] - Best Cost : [{cBest}]", end='\r', flush=True)
        return cBest
    
    def cbest_single_tree_multi(self, treeVerticesList, nodeTowardList, iteration, print_debug=False):
        xSolnCost = [[node.cost + self.cost_line(node, nodeTowardList[ind]) for node in vertexList] for ind, vertexList in enumerate(treeVerticesList)]
        cBest = None
        xGoalBestIndex = None
        for index, sublist in enumerate(xSolnCost):
            if not sublist:
                continue
            sublistMin = min(sublist)
            if cBest is None or sublistMin < cBest:
                cBest = sublistMin
                xGoalBestIndex = index

        if xGoalBestIndex is not None:
            if cBest < self.cBestPrevious:
                self.perfMatrix["Cost Graph"].append((iteration, cBest))
                self.cBestPrevious = cBest
            if print_debug:
                print(f"Iteration : [{iteration}] - Best Cost : [{cBest}]", end='\r', flush=True)
            return cBest, xGoalBestIndex
        else:
            return np.inf, None

    def plot_2d_obstacle(self, axis):
        jointRange = np.linspace(-np.pi, np.pi, 360)
        collisionPoint = []
        for theta1 in jointRange:
            for theta2 in jointRange:
                config = Node(np.array([theta1, theta2]).reshape(2, 1))
                result = self.robotEnv.collision_check(config)
                if result is True:
                    collisionPoint.append([theta1, theta2])

        collisionPoint = np.array(collisionPoint)
        axis.plot(collisionPoint[:, 0], collisionPoint[:, 1], color='darkcyan', linewidth=0, marker='o', markerfacecolor='darkcyan', markersize=1.5)

    def plot_2d_single_tree(self, tree, axis):
        for vertex in tree:
            if vertex.parent == None:
                pass
            else:
                axis.plot([vertex.config[0], vertex.parent.config[0]], [vertex.config[1], vertex.parent.config[1]], color="darkgray")

    def plot_2d_node_in_tree(self, tree, axis):
        for vertex in tree:
            if vertex.parent == None:
                pass
            else:
                axis.plot([vertex.config[0]], [vertex.config[1]], color="blue", linewidth=0, marker='o', markerfacecolor='yellow')

    def plot_2d_dual_tree(self, tree1, tree2, axis):
        for vertex in tree1:
            if vertex.parent == None:
                pass
            else:
                axis.plot([vertex.config[0], vertex.parent.config[0]], [vertex.config[1], vertex.parent.config[1]], color="darkgray")

        for vertex in tree2:
            if vertex.parent == None:
                pass
            else:
                axis.plot([vertex.config[0], vertex.parent.config[0]], [vertex.config[1], vertex.parent.config[1]], color="darkgray")

    def plot_2d_state_configuration(self, xStart, xApp, xGoal, axis):
        axis.plot(xStart.config[0], xStart.config[1], color="blue", linewidth=0, marker='o', markerfacecolor='yellow')

        if isinstance(xApp, list):
            for xA in xApp:
                axis.plot(xA.config[0], xA.config[1], color="blue", linewidth=0, marker='o', markerfacecolor='green')
            for xG in xGoal:
                axis.plot(xG.config[0], xG.config[1], color="blue", linewidth=0, marker='o', markerfacecolor='red')
        else:
            axis.plot(xApp.config[0], xApp.config[1], color="blue", linewidth=0, marker='o', markerfacecolor='green')
            axis.plot(xGoal.config[0], xGoal.config[1], color="blue", linewidth=0, marker='o', markerfacecolor='red')

    def plot_2d_path(self, path, axis):
        axis.plot([node.config[0] for node in path], [node.config[1] for node in path], color='blue', linewidth=2, marker='o', markerfacecolor='plum', markersize=5)

    def plot_performance(self, axis):
        costGraph = self.perfMatrix["Cost Graph"]
        iteration, costs = zip(*costGraph)

        legendItems = [
            mpatches.Patch(color='blue', label=f'Parameters: eta = [{self.perfMatrix["Parameters"]["eta"]}]'),
            mpatches.Patch(color='blue', label=f'Parameters: subEta = [{self.perfMatrix["Parameters"]["subEta"]}]'),
            mpatches.Patch(color='blue', label=f'Parameters: Max Iteration = [{self.perfMatrix["Parameters"]["Max Iteration"]}]'),
            mpatches.Patch(color='blue', label=f'Parameters: Rewire Radius = [{self.perfMatrix["Parameters"]["Rewire Radius"]}]'),
            mpatches.Patch(color='red', label=f'# Node = [{self.perfMatrix["Number of Node"]}]'),
            mpatches.Patch(color='green', label=f'Initial Path Cost = [{self.perfMatrix["Cost Graph"][0][1]:.5f}]'),
            mpatches.Patch(color='yellow', label=f'Initial Path Found on Iteration = [{self.perfMatrix["Cost Graph"][0][0]}]'),
            mpatches.Patch(color='pink', label=f'Final Path Cost = [{self.perfMatrix["Cost Graph"][-1][1]:.5f}]'),
            mpatches.Patch(color='indigo', label=f'Total Planning Time = [{self.perfMatrix["Total Planning Time"]:.5f}]'),
            mpatches.Patch(color='tan', label=f'Planning Time Only = [{self.perfMatrix["Planning Time Only"]:.5f}]'),
            mpatches.Patch(color='olive', label=f'KCD Time Spend = [{self.perfMatrix["KCD Time Spend"]:.5f}]'),
            mpatches.Patch(color='cyan', label=f'# KCD = [{self.perfMatrix["Number of Collision Check"]}]'),
            mpatches.Patch(color='peru', label=f'Avg KCD Time = [{self.perfMatrix["Average KCD Time"]:.5f}]'),
        ]

        axis.plot(iteration, costs, color='blue', marker='o', markersize=5)
        axis.legend(handles=legendItems)

        axis.set_xlabel('Iteration')
        axis.set_ylabel('Cost')
        axis.set_title(f'Performance Plot of [{self.perfMatrix["Planner Name"]}], Environment [{self.robotEnv.__class__.__name__}], DoF [{self.DoF}]')

    def perf_matrix_update(self, tree1=None, tree2=None, timePlanningStart=0, timePlanningEnd=0): # time arg is in nanosec
        self.perfMatrix["Planner Name"] = f"{self.__class__.__name__}"
        if tree2 is None:
            self.perfMatrix["Number of Node"] = len(tree1)
        elif tree2 is not None:
            self.perfMatrix["Number of Node"] = len(tree1) + len(tree2)
        self.perfMatrix["Total Planning Time"] = (timePlanningEnd-timePlanningStart) * 1e-9
        self.perfMatrix["KCD Time Spend"] = self.perfMatrix["KCD Time Spend"] * 1e-9
        self.perfMatrix["Planning Time Only"] = self.perfMatrix["Total Planning Time"] - self.perfMatrix["KCD Time Spend"]
        self.perfMatrix["Average KCD Time"] = self.perfMatrix["KCD Time Spend"] / self.perfMatrix["Number of Collision Check"]
    
    @classmethod
    def catch_key_interrupt(self, mainFunction):
        def wrapper(*args):
            try:
                mainFunction(*args)
                print("Done")
            except KeyboardInterrupt:
                print("User End Process")
        return wrapper