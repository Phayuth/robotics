import time
import numpy as np


class Node:

    def __init__(self, config, parent=None, cost=0.0) -> None:
        self.config = config
        self.parent = parent
        self.child = []
        self.cost = cost

    def __repr__(self) -> str:
        return f'\nconfig = {self.config.T}, hasParent = {True if self.parent != None else False}, NumChild = {len(self.child)}'


class RRTComponent:

    def __init__(self, baseConfigFile):
        # simulator
        self.simulator = baseConfigFile["simulator"] # simulator class
        self.configLimit = np.array(self.simulator.configLimit)
        self.configDoF = self.simulator.configDoF

        # planner properties : general, some parameter must be set and some are optinal with default value
        self.eta = baseConfigFile["eta"]
        self.subEta = baseConfigFile["subEta"]
        self.maxIteration = baseConfigFile["maxIteration"]
        self.nearGoalRadius = nR if (nR:=baseConfigFile["nearGoalRadius"]) is not None else self.eta # if given as the value then use it, otherwise equal to eta
        self.rewireRadius = baseConfigFile["rewireRadius"]
        self.probabilityGoalBias = baseConfigFile.get("probabilityGoalBias", 0.05)    # When close to goal select goal, with this probability? default: 0.05, must modify my code
        self.maxGoalSample = baseConfigFile.get("maxGoalSample", 10)                  # Goal samples are only sampled until maxSampleCount() goals are in the tree, to prohibit duplicate goal states.
        self.kNNTopNearest = baseConfigFile.get("kNNTopNearest", 10)                  # search all neighbour but only return top nearest nighbour during near neighbour search, if None, return all
        self.discreteLimitNumSeg = baseConfigFile.get("discreteLimitNumSeg", 10)      # limited number of segment divide for collision check and in goal region check

        # planner properties : termination
        self.endIterationID = baseConfigFile.get("endIterationID", 1)                 # break condition 1:maxIteration exceeded, 2: first solution found, 3:cost drop different is low
        self.terminateNumSolutions = baseConfigFile.get("terminateNumSolutions", 5)   # terminate planning when number of solution found is equal to it
        self.iterationDiffConstant = baseConfigFile.get("iterationDiffConstant", 100) # if iteration different is lower than this constant, terminate loop
        self.costDiffConstant = baseConfigFile.get("costDiffConstant", 0.001)         # if cost different is lower than this constant, terminate loop

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

        # keep track cost and iteration
        self.cBestPrevious = np.inf
        self.cBestPreviousIteration = 0
        self.cBestNow = np.inf
        self.iterationNow = 0

        # misc, for improve unnessary computation
        self.jointIndex = range(self.configDoF)
        self.gammaFunction = {1: 1, 2: 1, 2.5: 1.32934, 3: 2, 4: 6, 5: 24, 6: 120}  # given DoF, apply gamma(DoF)

        # debug setting
        self.printDebug = baseConfigFile.get("printDebug", False)

    def uni_sampling(self) -> Node:
        config = np.random.uniform(low=self.configLimit[:, 0], high=self.configLimit[:, 1]).reshape(-1, 1)
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
        u = np.random.normal(0.0, 1.0, (self.configDoF + 2, 1))
        norm = np.linalg.norm(u)
        u = u / norm
        return u[:self.configDoF,:] #The first N coordinates are uniform in a unit N ball

    def rotation_to_world(self, xStart, xGoal): # C
        cMin = self.distance_between_config(xStart, xGoal)
        a1 = (xGoal.config - xStart.config) / cMin
        I1 = np.array([1.0] + [0.0] * (self.configDoF - 1)).reshape(1, -1)
        M = a1 @ I1
        U, _, V_T = np.linalg.svd(M, True, True)
        middleTerm = [1.0] * (self.configDoF - 1) + [np.linalg.det(U) * np.linalg.det(V_T.T)]
        return U @ np.diag(middleTerm) @ V_T

    def hyperellipsoid_axis_length(self, cMax, cMin):  # L
        r1 = cMax / 2
        ri = np.sqrt(cMax**2 - cMin**2) / 2
        diagTerm = [r1] + [ri] * (self.configDoF - 1)
        return np.diag(diagTerm)

    def local_path_sampling(self, anchorPath, localPath, NumSeg):  #expected given path [xinit, x1, x2, ..., xcandidateToxApp, xApp]
        gRand = np.random.randint(low=0, high=NumSeg)
        randDownPercentage = np.random.uniform(low=0.0, high=1.0)
        randAlongPercentage = np.random.uniform(low=0.0, high=1.0)

        if gRand == 0:  # start Gap
            downRandRight = self.steer_rand_percentage(localPath[1], anchorPath[1], randDownPercentage)
            xRand = self.steer_rand_percentage(localPath[0], downRandRight, randAlongPercentage)
        elif gRand == NumSeg - 1:  # end Gap
            downRandLeft = self.steer_rand_percentage(localPath[-2], anchorPath[-2], randDownPercentage)
            xRand = self.steer_rand_percentage(downRandLeft, localPath[-1], randAlongPercentage)
        else:  # mid Gap
            downRandLeft = self.steer_rand_percentage(localPath[gRand], anchorPath[gRand], randDownPercentage)
            downRandRight = self.steer_rand_percentage(localPath[gRand + 1], anchorPath[gRand + 1], randDownPercentage)
            xRand = self.steer_rand_percentage(downRandLeft, downRandRight, randAlongPercentage)

        return xRand

    def unit_nball_volume_measure(self): # The Lebesgue measure (i.e., "volume") of an n-dimensional ball with a unit radius.
        return (np.pi**(self.configDoF / 2)) / self.gammaFunction[(self.configDoF / 2) + 1]  # ziD

    def prolate_hyperspheroid_measure(self): # The Lebesgue measure (i.e., "volume") of an n-dimensional prolate hyperspheroid (a symmetric hyperellipse) given as the distance between the foci and the transverse diameter.
        pass

    def lebesgue_obstacle_free_measure(self):
        diff = np.diff(self.configLimit)
        return np.prod(diff)

    def calculate_rewire_radius(self, numVertex, rewireFactor=1.1):
        inverseDoF = 1.0 / self.configDoF
        gammaRRG = rewireFactor * 2.0 * ((1.0+inverseDoF) * (self.lebesgue_obstacle_free_measure() / self.unit_nball_volume_measure()))**(inverseDoF)
        return np.min([self.eta, gammaRRG * (np.log(numVertex) / numVertex)**(inverseDoF)])

    def nearest_node(self, treeVertices, xCheck, returnDistList=False):
        distListToxCheck = [self.distance_between_config(xCheck, x) for x in treeVertices]
        minIndex = np.argmin(distListToxCheck)
        xNearest = treeVertices[minIndex]
        if returnDistList:
            return xNearest, distListToxCheck
        else:
            return xNearest

    def steer_rand_percentage(self, xFrom, xTo, percentage):  # percentage only between [0.0 - 1.0]
        distI = self.distance_each_component_between_config(xFrom, xTo)
        newI = xFrom.config + percentage*distI
        xNew = Node(newI)
        return xNew

    def steer(self, xFrom, xTo, distance, returnIsReached=False):
        distI = self.distance_each_component_between_config(xFrom, xTo)
        dist = np.linalg.norm(distI)
        isReached = False
        if dist <= distance:
            xNew = Node(xTo.config)
            isReached = True
        else:
            dI = (distI/dist) * distance
            newI = xFrom.config + dI
            xNew = Node(newI)
        if returnIsReached:
            return xNew, isReached
        else:
            return xNew

    def near(self, treeVertices, xCheck, searchRadius=None, distListToxCheck=None):
        if searchRadius is None:
            searchRadius = self.calculate_rewire_radius(len(treeVertices))

        if distListToxCheck:
            distListToxCheck = np.array(distListToxCheck)
        else:
            distListToxCheck = np.array([self.distance_between_config(xCheck, vertex) for vertex in treeVertices])

        nearIndices = np.where(distListToxCheck<=searchRadius)[0]

        if self.kNNTopNearest:
            if len(nearIndices) < self.kNNTopNearest:
                return [treeVertices[item] for item in nearIndices], distListToxCheck[nearIndices]
            else:
                nearDistList = distListToxCheck[nearIndices]
                sortedIndicesDist = np.argsort(nearDistList)
                topNearIndices = nearIndices[sortedIndicesDist[:self.kNNTopNearest]]
                return [treeVertices[item] for item in topNearIndices], distListToxCheck[topNearIndices]
        else:
            return [treeVertices[item] for item in nearIndices], distListToxCheck[nearIndices]

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
        distI = self.distance_each_component_between_config(xCheckFrom, xCheckTo)
        dist = np.linalg.norm(distI)
        if NumSeg:
            NumSeg = NumSeg
        else:
            NumSeg = int(np.ceil(dist / self.subEta))
            if NumSeg > self.discreteLimitNumSeg:
                NumSeg = self.discreteLimitNumSeg
        rateI = distI / NumSeg
        for i in range(1, NumSeg):
            newI = xCheckFrom.config + rateI*i
            xNew = Node(newI)
            if self.is_config_in_region_of_config(xNew, xCenter, radius):
                return True
        return False

    def is_config_in_collision(self, xCheck):
        timeStartKCD = time.perf_counter_ns()
        result = self.simulator.collision_check(xCheck.config)
        timeEndKCD = time.perf_counter_ns()
        self.perfMatrix["KCD Time Spend"] += timeEndKCD - timeStartKCD
        self.perfMatrix["Number of Collision Check"] += 1
        return result

    def is_connect_config_in_collision(self, xFrom, xTo, NumSeg=None):
        distI = self.distance_each_component_between_config(xFrom, xTo)
        dist = np.linalg.norm(distI)
        if NumSeg:
            NumSeg = NumSeg
        else:
            NumSeg = int(np.ceil(dist / self.subEta))
            if NumSeg > self.discreteLimitNumSeg:
                NumSeg = self.discreteLimitNumSeg
        rateI = distI / NumSeg
        for i in range(1, NumSeg):
            newI = xFrom.config + rateI*i
            xTo = Node(newI)
            if self.is_config_in_collision(xTo):
                return True
        return False

    def is_config_in_joint_limit(self, xCheck):
        if np.all(self.configLimit[:, 0] < xCheck.config < self.configLimit[:, 1]):
            return True
        else:
            return False

    # def is_goal_candidate_dismissable(self, xGoal):
    #     if self.cBestNow < self.distance_between_config(self.xStart, xGoal) and self.cBestNow < self.current_cost_to_goal_node(xGoal):
    #         return True
    #     else:
    #         return False

    # def current_cost_to_goal_node(self, xGoal):
    #     raise NotImplementedError

    def distance_between_config(self, xFrom, xTo):
        return np.linalg.norm(self.distance_each_component_between_config(xFrom, xTo))

    def distance_each_component_between_config(self, xFrom, xTo):
        return xTo.config - xFrom.config

    def star_optimizer(self, treeToAdd, xNew, rewireRadius, xNewIsxRand=None, preCalDistListToxNew=None):
        # optimized to reduce dist cal if xNew is xRand
        if xNewIsxRand:
            XNear, XNearToxNewDist = self.near(treeToAdd, xNew, rewireRadius, preCalDistListToxNew)
        else:
            XNear, XNearToxNewDist = self.near(treeToAdd, xNew, rewireRadius)

        # Parenting
        cMin = xNew.cost
        # XNearCostToxNew = [xNear.cost + self.cost_line(xNear, xNew) for xNear in XNear] # original, here we exploit because cost is just a distance that we already calculate. if cost is anything else, then we use this line instead.
        XNearCostToxNew = [xNear.cost + XNearToxNewDist[i] for i, xNear in enumerate(XNear)]
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
            # cNew = xNew.cost + self.cost_line(xNew, xNear) # original
            cNew = xNew.cost + XNearToxNewDist[index]
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

    def get_children_until_depth(self, depth=1):
        children = []
        def get_children_recursive(node, depth):
            if depth is None:
                if len(node.child) != 0:
                    for nc in node.child:
                        children.append(nc)
                        get_children_recursive(nc, depth)
            elif depth is not None:
                if len(node.child) != 0 and depth!=0:
                    depth = -1
                    for nc in node.child:
                        children.append(node.child)
                        get_children_recursive(node.child, depth)

    def get_ancestor_until_depth(self, node, depth=1): # if depth is None : go all the ways to the root, otherwise : go until specific depth. depth = 0 is empty list, use depth = 1 in general
        ancestor = []
        def get_ancestor_recursive(node, depth):
            if depth is None: # go all the ways to the root
                if node.parent is not None:
                    ancestor.append(node.parent)
                    get_ancestor_recursive(node.parent, depth)
            elif depth is not None: # go until specific depth
                if node.parent is not None and depth!=0:
                    depth -= 1
                    ancestor.append(node.parent)
                    get_ancestor_recursive(node.parent, depth)

        get_ancestor_recursive(node, depth)
        return ancestor

    def near_ancestor(self, XNear, depth=1):
        nearAncestor = []
        for xNear in XNear:
            nearAncestor.extend(self.get_ancestor_until_depth(xNear, depth))
        nearAncestor = set(nearAncestor) # remove duplicate, ordered is lost
        nearAncestor = list(nearAncestor)
        return nearAncestor

    def star_quick_optimizer(self, treeToAdd, xNew, rewireRadius, xNewIsxRand=None, preCalDistListToxNew=None):
        # optimized to reduce dist cal if xNew is xRand
        if xNewIsxRand:
            XNear, _ = self.near(treeToAdd, xNew, rewireRadius, preCalDistListToxNew)
        else:
            XNear, _ = self.near(treeToAdd, xNew, rewireRadius)

        # rrt star quick
        XNearAncestor = self.near_ancestor(XNear, depth=1)
        if len(XNearAncestor) == 0:
            XNear = XNear
        else:
            XNear = XNear + XNearAncestor

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

    def termination_check(self, solutionList):
        if self.endIterationID == 1: # maxIteration exceeded
            return self.termination_on_max_iteration()
        elif self.endIterationID == 2: # first solution found
            return self.termination_on_first_k_solution(solutionList)
        elif self.endIterationID == 3: # cost drop different is low
            return self.termination_on_cost_drop_different()
        elif self.endIterationID == 4: # cost and iteration drop different is low
            return self.termination_on_cost_iteration_drop_different()

    def termination_on_max_iteration(self): # already in for loop, no break is needed
        return False

    def termination_on_first_k_solution(self, solutionList):
        if len(solutionList) == self.terminateNumSolutions:
            return True
        else:
            return False

    def termination_on_cost_drop_different(self):
        if self.cBestPrevious - self.cBestNow < self.costDiffConstant:
            return True
        else:
            return False

    def termination_on_cost_iteration_drop_different(self):
        if (self.cBestPrevious - self.cBestNow < self.costDiffConstant) and (self.iterationNow - self.cBestPreviousIteration > self.iterationDiffConstant):
            return True
        else:
            return False

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

    def cbest_single_tree(self, treeVertices, nodeToward, iteration): # search in treeGoalRegion for the current best cost
        self.iterationNow = iteration
        if len(treeVertices) == 0:
            cBest = np.inf
            xSolnCost = []
        else:
            xSolnCost = [xSoln.cost + self.cost_line(xSoln, nodeToward) for xSoln in treeVertices]
            cBest = min(xSolnCost)
            # self.perfMatrix["Cost Graph"].append((iteration, cBest))
            if cBest < self.cBestPrevious : # this has nothing to do with planning itself, just for record performance data only
                self.perfMatrix["Cost Graph"].append((iteration, cBest))
                self.cBestPrevious = cBest
                self.cBestPreviousIteration = iteration
        if self.printDebug:
            print(f"Iteration : [{iteration}] - Best Cost : [{cBest}]", end='\r', flush=True)
        return cBest

    def cbest_dual_tree(self, connectNodePair, iteration): # search in connectNodePairList for the current best cost
        self.iterationNow = iteration
        if len(connectNodePair) == 0:
            cBest = np.inf
            xSolnCost = []
        else:
            xSolnCost = [vertexA.cost + vertexB.cost + self.cost_line(vertexA, vertexB) for vertexA, vertexB in connectNodePair]
            cBest = min(xSolnCost)
            # self.perfMatrix["Cost Graph"].append((iteration, cBest))
            if cBest < self.cBestPrevious:
                self.perfMatrix["Cost Graph"].append((iteration, cBest))
                self.cBestPrevious = cBest
                self.cBestPreviousIteration = iteration
        if self.printDebug:
            print(f"Iteration : [{iteration}] - Best Cost : [{cBest}]", end='\r', flush=True)
        return cBest

    def cbest_single_tree_multi(self, treeVerticesList, nodeTowardList, iteration):
        self.iterationNow = iteration
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
            # self.perfMatrix["Cost Graph"].append((iteration, cBest))
            if cBest < self.cBestPrevious:
                self.perfMatrix["Cost Graph"].append((iteration, cBest))
                self.cBestPrevious = cBest
                self.cBestPreviousIteration = iteration
            if self.printDebug:
                print(f"Iteration : [{iteration}] - Best Cost : [{cBest}]", end='\r', flush=True)
            return cBest, xGoalBestIndex
        else:
            return np.inf, None

    def update_perf(self, timePlanningStart=0, timePlanningEnd=0): # time arg is in nanosec
        self.perfMatrix["Planner Name"] = f"{self.__class__.__name__}"
        if hasattr(self, "treeVertex"):
            self.perfMatrix["Number of Node"] = len(getattr(self, "treeVertex")) # list
        elif hasattr(self, "treeVertexStart") and hasattr(self, "treeVertexGoal"):
            self.perfMatrix["Number of Node"] = len(getattr(self, "treeVertexStart")) + len(getattr(self, "treeVertexGoal"))
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

    def start(self):
        raise NotImplementedError("Individual planner must implement its own start() method")

    def get_path(self): # list of nodes sequence
        raise NotImplementedError("Individual planner must implement its own get_path() method")

    def get_path_array(self, pathNodesSeq):
        pathTemp = [node.config for node in pathNodesSeq]
        return np.hstack(pathTemp) # shape=(numDoF, numSeq)

    def begin_planner(self):
        timeStart = time.perf_counter_ns()
        self.start()
        path = self.get_path()
        timeEnd = time.perf_counter_ns()
        self.update_perf(timeStart, timeEnd)

        if path is not None: # there are solutions
            return self.get_path_array(path)
        else: # no solution
            return None

    # experimental
    def uni_sampling_pseudo(self, tree):
        # no k-nn is calculated, the actual approach would mostly get extend fast at early iteration since it will mostly over eta
        # not working as expected
        gRand = np.random.randint(0, len(tree))
        vRand = np.random.uniform(-1, 1, (self.configDoF, 1))
        dRand = (vRand/np.linalg.norm(vRand))*self.eta
        xGrowFrom = tree[gRand]
        newI = xGrowFrom.config + dRand
        xNew = Node(newI)
        return xNew, xGrowFrom

    def nearest_pseudo(self, tree):
        pass