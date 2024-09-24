import os
import sys

sys.path.append(str(os.path.abspath(os.getcwd())))

import time
import numpy as np
from spatial_geometry.utils import Utils
import matplotlib.patches as mpatches


class Node:

    def __init__(self, config, parent=None, cost=0.0) -> None:
        self.config = config
        self.parent = parent
        self.child = []
        self.cost = cost

    def __repr__(self) -> str:
        return f"\nconfig = {self.config.T}, hasParent = {True if self.parent != None else False}, NumChild = {len(self.child)}"


class RRTTorusRedundantComponent:

    def __init__(self, baseConfigFile):
        # simulator
        self.simulator = baseConfigFile["simulator"]  # simulator class
        self.configLimit = np.array(self.simulator.configLimit)
        self.configDoF = self.simulator.configDoF
        self.lim = np.array([[-2 * np.pi, 2 * np.pi] * self.configDoF])

        # planner properties : general, some parameter must be set and some are optinal with default value
        self.eta = baseConfigFile["eta"]
        self.subEta = baseConfigFile["subEta"]
        self.maxIteration = baseConfigFile["maxIteration"]
        self.nearGoalRadius = nR if (nR := baseConfigFile["nearGoalRadius"]) is not None else self.eta  # if given as the value then use it, otherwise equal to eta
        self.rewireRadius = baseConfigFile["rewireRadius"]
        self.probabilityGoalBias = baseConfigFile.get("probabilityGoalBias", 0.05)  # When close to goal select goal, with this probability? default: 0.05, must modify my code
        self.maxGoalSample = baseConfigFile.get("maxGoalSample", 10)  # Goal samples are only sampled until maxSampleCount() goals are in the tree, to prohibit duplicate goal states.
        self.kNNTopNearest = baseConfigFile.get("kNNTopNearest", 10)  # search all neighbour but only return top nearest nighbour during near neighbour search, if None, return all
        self.discreteLimitNumSeg = baseConfigFile.get("discreteLimitNumSeg", 10)  # limited number of segment divide for collision check and in goal region check

        # planner properties : termination
        self.endIterationID = baseConfigFile.get("endIterationID", 1)  # break condition 1:maxIteration exceeded, 2: first solution found, 3:cost drop different is low
        self.terminateNumSolutions = baseConfigFile.get("terminateNumSolutions", 5)  # terminate planning when number of solution found is equal to it
        self.iterationDiffConstant = baseConfigFile.get("iterationDiffConstant", 100)  # if iteration different is lower than this constant, terminate loop
        self.costDiffConstant = baseConfigFile.get("costDiffConstant", 0.001)  # if cost different is lower than this constant, terminate loop

        # collision database
        self.configSearched = []
        self.collisionState = []

        # performance matrix
        self.perfMatrix = {
            "Planner Name": "",
            "Parameters": {"eta": self.eta, "subEta": self.subEta, "Max Iteration": self.maxIteration, "Rewire Radius": self.rewireRadius},
            "Number of Node": 0,
            "Total Planning Time": 0.0,  # include KCD
            "KCD Time Spend": 0.0,
            "Planning Time Only": 0.0,
            "Number of Collision Check": 0,
            "Average KCD Time": 0.0,
            "Cost Graph": [],
        }

        # keep track cost and iteration
        self.cBestPrevious = np.inf
        self.cBestPreviousIteration = 0
        self.cBestNow = np.inf
        self.iterationNow = 0

        # misc, for improve unnessary computation
        self.jointIndex = range(self.configDoF)

        # debug setting
        self.printDebug = baseConfigFile.get("printDebug", False)

    def uni_sampling(self) -> Node:
        config = np.random.uniform(low=self.configLimit[:, 0], high=self.configLimit[:, 1]).reshape(-1, 1)
        xRand = Node(config)
        return xRand

    def nearest_wrap_node(self, treeVertices: list[Node], xCheck: Node):
        dist = [Utils.minimum_dist_torus(xCheck.config, xi.config) for xi in treeVertices]
        minIndex = np.argmin(dist)
        return treeVertices[minIndex]

    def near_wrap(self, treeVertices, xCheck, searchRadius=None):
        if searchRadius is None:
            searchRadius = self.eta

        distListToxCheck = np.array([Utils.minimum_dist_torus(xCheck.config, vertex.config) for vertex in treeVertices])
        nearIndices = np.where(distListToxCheck <= searchRadius)[0]
        return [treeVertices[item] for item in nearIndices], distListToxCheck[nearIndices]

    def steer(self, xFrom, xTo, distance):
        distI = xTo - xFrom
        dist = np.linalg.norm(distI)
        if dist <= distance:
            xNew = xTo
        else:
            dI = (distI / dist) * distance
            newI = xFrom + dI
            xNew = newI
        return xNew

    def steer_wrap(self, xFrom, xTo, distance):
        candi = Utils.nearest_qb_to_qa(xFrom.config, xTo.config, self.lim, ignoreOrginal=False)
        newI = self.steer(xFrom.config, candi, distance)
        return Node(Utils.wrap_to_pi(newI))

    def cost_line(self, xFrom, xTo):
        return Utils.minimum_dist_torus(xFrom.config, xTo.config)

    def is_collision(self, xFrom, xTo):
        if self.is_config_in_collision(xTo):
            return True
        elif self.is_connect_config_in_collision(xFrom, xTo):
            return True
        else:
            return False

    def is_config_in_collision(self, xCheck):
        timeStartKCD = time.perf_counter_ns()
        result = self.simulator.collision_check(xCheck.config)
        timeEndKCD = time.perf_counter_ns()
        self.perfMatrix["KCD Time Spend"] += timeEndKCD - timeStartKCD
        self.perfMatrix["Number of Collision Check"] += 1
        return result

    def is_connect_config_in_collision(self, xFrom, xTo, NumSeg=None):
        distI = Utils.nearest_qb_to_qa(xFrom.config, xTo.config, self.lim, ignoreOrginal=False) - xFrom.config
        dist = np.linalg.norm(distI)
        if NumSeg:
            NumSeg = NumSeg
        else:
            NumSeg = int(np.ceil(dist / self.subEta))
            if NumSeg > self.discreteLimitNumSeg:
                NumSeg = self.discreteLimitNumSeg
        rateI = distI / NumSeg
        for i in range(1, NumSeg):
            newI = xFrom.config + rateI * i
            xTo = Node(newI)
            if self.is_config_in_collision(xTo):
                return True
        return False

    def is_config_in_region_of_config(self, xCheck, xCenter, radius):
        if Utils.minimum_dist_torus(xCheck.config, xCenter.config) < radius:
            return True
        return False

    def star_optimizer(self, treeToAdd, xNew, rewireRadius):
        # optimized to reduce dist cal if xNew is xRand
        XNear, XNearToxNewDist = self.near_wrap(treeToAdd, xNew, rewireRadius)

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

    def update_child_cost(self, xCheck, treeToAdd=None):  # recursively updates the cost of the children of this node if the cost up to this node has changed.
        for child in xCheck.child:
            child.cost = child.parent.cost + self.cost_line(child.parent, child)
            if treeToAdd:
                treeToAdd.append(child)
            self.update_child_cost(child)

    def termination_check(self, solutionList):
        if self.endIterationID == 1:  # maxIteration exceeded
            return self.termination_on_max_iteration()
        elif self.endIterationID == 2:  # first solution found
            return self.termination_on_first_k_solution(solutionList)
        elif self.endIterationID == 3:  # cost drop different is low
            return self.termination_on_cost_drop_different()
        elif self.endIterationID == 4:  # cost and iteration drop different is low
            return self.termination_on_cost_iteration_drop_different()

    def termination_on_max_iteration(self):  # already in for loop, no break is needed
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

    def cbest_single_tree(self, treeVertices, nodeToward, iteration):  # search in treeGoalRegion for the current best cost
        self.iterationNow = iteration
        if len(treeVertices) == 0:
            cBest = np.inf
            xSolnCost = []
        else:
            xSolnCost = [xSoln.cost + self.cost_line(xSoln, nodeToward) for xSoln in treeVertices]
            cBest = min(xSolnCost)
            # self.perfMatrix["Cost Graph"].append((iteration, cBest))
            if cBest < self.cBestPrevious:  # this has nothing to do with planning itself, just for record performance data only
                self.perfMatrix["Cost Graph"].append((iteration, cBest))
                self.cBestPrevious = cBest
                self.cBestPreviousIteration = iteration
        if self.printDebug:
            print(f"Iteration : [{iteration}] - Best Cost : [{cBest}]", end="\r", flush=True)
        return cBest

    def update_perf(self, timePlanningStart=0, timePlanningEnd=0):  # time arg is in nanosec
        self.perfMatrix["Planner Name"] = f"{self.__class__.__name__}"
        if hasattr(self, "treeVertex"):
            self.perfMatrix["Number of Node"] = len(getattr(self, "treeVertex"))  # list
        elif hasattr(self, "treeVertexStart") and hasattr(self, "treeVertexGoal"):
            self.perfMatrix["Number of Node"] = len(getattr(self, "treeVertexStart")) + len(getattr(self, "treeVertexGoal"))
        self.perfMatrix["Total Planning Time"] = (timePlanningEnd - timePlanningStart) * 1e-9
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

    def begin_planner(self):
        timeStart = time.perf_counter_ns()
        self.start()
        path = self.get_path()
        timeEnd = time.perf_counter_ns()
        self.update_perf(timeStart, timeEnd)

        return path


class RRTPlotter:

    def plot_2d_obstacle(simulator, axis):
        joint1Range = np.linspace(simulator.configLimit[0][0], simulator.configLimit[0][1], 360)
        joint2Range = np.linspace(simulator.configLimit[1][0], simulator.configLimit[1][1], 360)
        collisionPoint = []
        for theta1 in joint1Range:
            for theta2 in joint2Range:
                config = np.array([[theta1], [theta2]])
                result = simulator.collision_check(config)
                if result is True:
                    collisionPoint.append([theta1, theta2])

        collisionPoint = np.array(collisionPoint)
        axis.plot(collisionPoint[:, 0], collisionPoint[:, 1], color="darkcyan", linewidth=0, marker="o", markerfacecolor="darkcyan", markersize=1.5)

    def plot_2d_tree(trees, axis):
        limt2 = np.array([[-2 * np.pi, 2 * np.pi], [-2 * np.pi, 2 * np.pi]])
        for tree in trees:
            for vertex in tree:
                if vertex.parent == None:
                    pass
                else:
                    # axis.plot([vertex.config[0], vertex.parent.config[0]], [vertex.config[1], vertex.parent.config[1]], color="darkgray")
                    qabw = Utils.nearest_qb_to_qa(vertex.config, vertex.parent.config, limt2, ignoreOrginal=False)
                    qbaw = Utils.nearest_qb_to_qa(vertex.parent.config, vertex.config, limt2, ignoreOrginal=False)
                    axis.plot([vertex.config[0], qabw[0]], [vertex.config[1], qabw[1]], color="darkgray", marker="o", markerfacecolor="black")
                    axis.plot([vertex.parent.config[0], qbaw[0]], [vertex.parent.config[1], qbaw[1]], color="darkgray", marker="o", markerfacecolor="black")

    def plot_2d_state(xStart, xApp, xGoal, axis):
        axis.plot(xStart.config[0], xStart.config[1], color="blue", linewidth=0, marker="o", markerfacecolor="yellow")

        if isinstance(xApp, list):
            for xA in xApp:
                axis.plot(xA.config[0], xA.config[1], color="blue", linewidth=0, marker="o", markerfacecolor="green")
            for xG in xGoal:
                axis.plot(xG.config[0], xG.config[1], color="blue", linewidth=0, marker="o", markerfacecolor="red")
        else:
            axis.plot(xApp.config[0], xApp.config[1], color="blue", linewidth=0, marker="o", markerfacecolor="green")
            axis.plot(xGoal.config[0], xGoal.config[1], color="blue", linewidth=0, marker="o", markerfacecolor="red")

    def plot_2d_path(path, axis):
        limt2 = np.array([[-2 * np.pi, 2 * np.pi], [-2 * np.pi, 2 * np.pi]])
        pc = path[0].config
        for i in range(1, len(path)):
            qabw = Utils.nearest_qb_to_qa(pc, path[i].config, limt2, ignoreOrginal=False)
            qbaw = Utils.nearest_qb_to_qa(path[i].config, pc, limt2, ignoreOrginal=False)
            axis.plot([pc[0], qabw[0]], [pc[1], qabw[1]], color="blue", linewidth=2, marker="o", markerfacecolor="plum", markersize=5)
            axis.plot([path[i].config[0], qbaw[0]], [path[i].config[1], qbaw[1]], color="blue", linewidth=2, marker="o", markerfacecolor="plum", markersize=5)
            pc = path[i].config

    def plot_2d_complete(plannerClass, path, ax):
        RRTPlotter.plot_2d_obstacle(plannerClass.simulator, ax)

        try:
            RRTPlotter.plot_2d_tree([plannerClass.treeVertexStart, plannerClass.treeVertexGoal], ax)
        except:
            RRTPlotter.plot_2d_tree([plannerClass.treeVertex], ax)

        if path is not None:
            RRTPlotter.plot_2d_path(path, ax)
        try:
            RRTPlotter.plot_2d_state(plannerClass.xStart, plannerClass.xApp, plannerClass.xGoal, ax)
        except:
            RRTPlotter.plot_2d_state(plannerClass.xStart, plannerClass.xAppList, plannerClass.xGoalList, ax)

    def plot_performance(perfMatrix, axis):
        costGraph = perfMatrix["Cost Graph"]
        iteration, costs = zip(*costGraph)

        legendItems = [
            mpatches.Patch(color="blue", label=f'Parameters: eta = [{perfMatrix["Parameters"]["eta"]}]'),
            mpatches.Patch(color="blue", label=f'Parameters: subEta = [{perfMatrix["Parameters"]["subEta"]}]'),
            mpatches.Patch(color="blue", label=f'Parameters: Max Iteration = [{perfMatrix["Parameters"]["Max Iteration"]}]'),
            mpatches.Patch(color="blue", label=f'Parameters: Rewire Radius = [{perfMatrix["Parameters"]["Rewire Radius"]}]'),
            mpatches.Patch(color="red", label=f'# Node = [{perfMatrix["Number of Node"]}]'),
            mpatches.Patch(color="green", label=f'Initial Path Cost = [{perfMatrix["Cost Graph"][0][1]:.5f}]'),
            mpatches.Patch(color="yellow", label=f'Initial Path Found on Iteration = [{perfMatrix["Cost Graph"][0][0]}]'),
            mpatches.Patch(color="pink", label=f'Final Path Cost = [{perfMatrix["Cost Graph"][-1][1]:.5f}]'),
            mpatches.Patch(color="indigo", label=f'Total Planning Time = [{perfMatrix["Total Planning Time"]:.5f}]'),
            mpatches.Patch(color="tan", label=f'Planning Time Only = [{perfMatrix["Planning Time Only"]:.5f}]'),
            mpatches.Patch(color="olive", label=f'KCD Time Spend = [{perfMatrix["KCD Time Spend"]:.5f}]'),
            mpatches.Patch(color="cyan", label=f'# KCD = [{perfMatrix["Number of Collision Check"]}]'),
            mpatches.Patch(color="peru", label=f'Avg KCD Time = [{perfMatrix["Average KCD Time"]:.5f}]'),
        ]

        axis.plot(iteration, costs, color="blue", marker="o", markersize=5)
        axis.legend(handles=legendItems)

        axis.set_xlabel("Iteration")
        axis.set_ylabel("Cost")
        axis.set_title(f'Performance Plot of [{perfMatrix["Planner Name"]}]')