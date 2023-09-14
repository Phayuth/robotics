import os
import sys
wd = os.path.abspath(os.getcwd())
sys.path.append(str(wd))

import time
import numpy as np
from copsim.arm_api import UR5eVirtualArmCoppeliaSimAPI

class Node:

    def __init__(self, x, y, z, p, q, r, parent=None, cost=0.0) -> None:
        self.x = x
        self.y = y
        self.z = z
        self.p = p
        self.q = q
        self.r = r
        self.parent = parent
        self.child = []
        self.cost = cost

    def __repr__(self) -> str:
        return f'\nconfig = [{self.x:.7f}, {self.y:.7f}, {self.z:.7f}, {self.p:.7f}, {self.q:.7f}, {self.r:.7f}, hasParent = {True if self.parent != None else False}, NumChild = {len(self.child)}]'

class CopSimRRTComponent():

    def __init__(self) -> None:
        # coppeliasim
        self.copHandle = UR5eVirtualArmCoppeliaSimAPI()

        # joint limit
        self.xMinRange = -np.pi
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

        # planner properties
        self.probabilityGoalBias = 0.2

        # collision database
        self.configSearched = []
        self.collisionState = []

        # performance matrix
        self.perfMatrix = {
            "Planner Name": "",
            "Parameters": {"eta": 0.0, "subEta":0.0, "Max Iteration": 0, "Rewire Radius": 0.0},
            "Number of Node": 0,
            "Total Planning Time": 0.0,  # include KCD
            "KCD Time Spend": 0.0,
            "Planning Time Only": 0.0,
            "Number of Collision Check": 0,
            "Average KCD Time": 0.0,
            "Cost Graph": []
        }

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
            xRand = Node(biasTowardNode.x, 
                         biasTowardNode.y, 
                         biasTowardNode.z, 
                         biasTowardNode.p, 
                         biasTowardNode.q, 
                         biasTowardNode.r)
        else:
            xRand = self.uni_sampling()
        return xRand

    def nearest_node(self, treeVertex, xRand):
        vertexList = []

        for eachVertex in treeVertex:
            vertexList.append(self.distance_between_config(xRand, eachVertex))

        minIndex = np.argmin(vertexList)
        xNear = treeVertex[minIndex]

        return xNear

    def steer(self, xNearest, xRand, toDistance):
        distX, distY, distZ, distP, distQ, distR = self.distance_each_component_between_config(xNearest, xRand)
        dist = np.linalg.norm([distX, distY, distZ, distP, distQ, distR])
        if dist <= toDistance:
            xNew = Node(xRand.x, xRand.y, xRand.z, xRand.p, xRand.q, xRand.r)
        else:
            dX = (distX/dist) * toDistance
            dY = (distY/dist) * toDistance
            dZ = (distZ/dist) * toDistance
            dP = (distP/dist) * toDistance
            dQ = (distQ/dist) * toDistance
            dR = (distR/dist) * toDistance
            newX = xNearest.x + dX
            newY = xNearest.y + dY
            newZ = xNearest.z + dZ
            newP = xNearest.p + dP
            newQ = xNearest.q + dQ
            newR = xNearest.r + dR
            xNew = Node(newX, newY, newZ, newP, newQ, newR)

        return xNew

    def near_in_radius(self, treeToSearch, xNew, radiusToSearch):
        # original Near Algorithm from theory but not practical in realworld
        neighbor = []
        for index, vertex in enumerate(treeToSearch):
            dist = np.linalg.norm([(xNew.x - vertex.x),
                                    (xNew.y - vertex.y),
                                    (xNew.z - vertex.z),
                                    (xNew.p - vertex.p),
                                    (xNew.q - vertex.q),
                                    (xNew.r - vertex.r)])
            if dist <= radiusToSearch:
                neighbor.append(index)
        return [treeToSearch[i] for i in neighbor]

    def near(self, treeToSearch, xNew, radiusToSearch):
        # fixed subset of radius for rewire, to avoid too much rewiring as the original algorithm
        # option : pick the nearest #numTopNearest node nearest to xNew 
        numTopNearest = 10
        neighborIndexandDist = []
        for index, vertex in enumerate(treeToSearch):
            dist = np.linalg.norm([(xNew.x - vertex.x),
                                    (xNew.y - vertex.y),
                                    (xNew.z - vertex.z),
                                    (xNew.p - vertex.p),
                                    (xNew.q - vertex.q),
                                    (xNew.r - vertex.r)])
            if dist <= radiusToSearch:
                neighborIndexandDist.append([index, dist])
        
        if len(neighborIndexandDist) < numTopNearest:
            return [treeToSearch[i] for i in [item[0] for item in neighborIndexandDist]]
        else:
            sortedDist = sorted(neighborIndexandDist, key=lambda x: x[1])
            indexSorted = [sortedDist[j][0] for j in [i for i in range(numTopNearest)]]
            return [treeToSearch[i] for i in indexSorted]
    
    def cost_line(self, xStart, xEnd):
        return self.distance_between_config(xStart, xEnd)

    def is_collision_and_in_goal_region(self, xNearest, xNew, xCenter, radius): # return True if any of it is true
        if self.is_config_in_region_of_config(xNew, xCenter, radius):
            return True
        elif self.is_connect_config_in_region_of_config(xNearest, xNew, xCenter, radius):
            return True
        elif self.is_config_in_collision(xNew):
            return True
        elif self.is_connect_config_in_collision(xNearest, xNew):
            return True
        else:
            return False

    def is_config_in_region_of_config(self, xToCheck, xCenter, radius):
        if self.distance_between_config(xToCheck, xCenter) < radius:
            return True
        return False

    def is_connect_config_in_region_of_config(self, xToCheckStart, xToCheckEnd, xCenter, radius, NumSeg=10):
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
        timeStartKCD = time.perf_counter_ns()
        result = self.copHandle.collsion_check(xNew)
        timeEndKCD = time.perf_counter_ns()
        self.perfMatrix["KCD Time Spend"] += timeEndKCD - timeStartKCD
        self.perfMatrix["Number of Collision Check"] += 1
        return result

    def is_connect_config_in_collision_fixed_nseg(self, xNearest, xNew, NumSeg=10):
        # fixed number of Segmented line
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

    def is_connect_config_in_collision(self, xNearest, xNew):
        # dynamic number of Segmented line, but still limit to 10 to avoid long check
        limitedNumSeg = 10
        distX, distY, distZ, distP, distQ, distR = self.distance_each_component_between_config(xNearest, xNew)
        dist = np.linalg.norm([distX, distY, distZ, distP, distQ, distR])
        NumSeg = int(np.ceil(dist/self.subEta))
        if NumSeg > limitedNumSeg:
            NumSeg = limitedNumSeg
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
    
    def tree_swap_flag(self):
        if self.treeSwapFlag is True:
            self.treeSwapFlag = False
        elif self.treeSwapFlag is False:
            self.treeSwapFlag = True

    def search_backtrack_single_directional_path(self, backFromNode, attachNode): # return path is [xinit, x1, x2, ..., xapp, xgoal]
        pathStart = [backFromNode]
        currentNodeStart = backFromNode
        while currentNodeStart.parent is not None:
            currentNodeStart = currentNodeStart.parent
            pathStart.append(currentNodeStart)

        pathStart.reverse()

        if attachNode:
            return pathStart + [attachNode] # attachNode is Normally xGoal
        else:
            return pathStart

    def search_backtrack_bidirectional_path(self, backFromNodeTa, backFromNodeTb, attachNode): # return path is [xinit, x1, x2, ..., xapp, xgoal]
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
            return pathStart + pathGoal + [attachNode] # attachNode is Normally xGoal
        else:
            return pathStart + pathGoal
    
    def search_best_cost_singledirection_path(self, backFromNode, treeVertexList, attachNode):
        vertexListCost = [vertex.cost for vertex in treeVertexList]
        costSortedIndex = sorted(range(len(vertexListCost)), key=lambda i: vertexListCost[i])

        for xNearIndex in costSortedIndex:
            if not self.is_connect_config_in_collision(treeVertexList[xNearIndex], backFromNode):
                path = [backFromNode]
                currentNodeStart = treeVertexList[xNearIndex]
                while currentNodeStart.parent is not None:
                    currentNodeStart = currentNodeStart.parent
                    path.append(currentNodeStart)
                print(path)
                path.reverse()

                if attachNode:
                    return path + [attachNode] # attachNode is Normally xGoal
                else:
                    return path
    
    # optimization code
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
