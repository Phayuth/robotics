import time
import numpy as np


class Node:

    def __init__(self, config=None, name=None) -> None:
        self.name = name
        self.config = config
        self.cost = np.inf
        self.pathvia = None
        self.edgeNodes = []
        self.edgeCosts = []

    def __repr__(self) -> str:
        return f"Node:{self.name}"


class AStar:

    def __init__(self, graph) -> None:
        self.graph = graph
        self.heapq = graph
        self.visitedNode = []

    def heapq_prio_heuristic(self, goal: Node):  # there are some method better than this but i want to test myself
        costs = np.array([hg.cost for hg in self.heapq])
        coststogo = np.array([self.cost_to_go(goal, hg) for hg in self.heapq])
        ci = np.argsort(costs + coststogo)
        self.heapq = [self.heapq[i] for i in ci]

    def search(self, start: Node, goal: Node):
        # set start at 0
        start.cost = 0

        while True:
            if len(self.visitedNode) == len(self.graph):
                print("no path were found")
                return

            self.heapq_prio_heuristic(goal)
            currentNode = self.heapq[0]

            if currentNode is goal:
                return self.backtrack(goal)

            for ei, ed in enumerate(currentNode.edgeNodes):
                if ed in self.visitedNode:
                    continue
                if mind := currentNode.cost + currentNode.edgeCosts[ei] < ed.cost:
                    ed.cost = mind
                    ed.pathvia = currentNode
            self.visitedNode.append(currentNode)
            self.heapq.pop(0)

    def cost_to_go(self, xTo, xFrom):  # euclidean distance
        return np.linalg.norm(xTo.config - xFrom.config)

    def backtrack(self, node: Node):
        path = [node]
        current = node
        while current.pathvia is not None:
            path.append(current.pathvia)
            current = current.pathvia
        return path
