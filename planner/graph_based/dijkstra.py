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


class Dijkstra:

    def __init__(self, graph) -> None:
        self.graph = graph
        self.heapq = graph
        self.visitedNode = []

    def heapq_prio(self):  # there are some method better than this but i want to test myself
        costs = [hg.cost for hg in self.heapq]
        ci = np.argsort(costs)
        self.heapq = [self.heapq[i] for i in ci]

    def search(self, start: Node, goal: Node):
        # set start at 0
        start.cost = 0

        while True:
            if len(self.visitedNode) == len(self.graph):
                print("no path were found")
                return

            self.heapq_prio()
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

    def backtrack(self, node: Node):
        path = [node]
        current = node
        while current.pathvia is not None:
            path.append(current.pathvia)
            current = current.pathvia
        return path


if __name__ == "__main__":
    # create node
    a = Node("a")
    b = Node("b")
    c = Node("c")
    d = Node("d")
    e = Node("e")
    f = Node("f")
    g = Node("g")
    h = Node("h")
    i = Node("i")

    # add edge
    a.edgeNodes = [b, c]
    a.edgeCosts = [2, 3]

    b.edgeNodes = [a, d]
    b.edgeCosts = [2, 7]
    c.edgeNodes = [a, d]
    c.edgeCosts = [3, 5]

    d.edgeNodes = [b, c, e, f]
    d.edgeCosts = [7, 5, 3, 6]

    e.edgeNodes = [d, f, g]
    e.edgeCosts = [3, 2, 3]
    f.edgeNodes = [d, e, h]
    f.edgeCosts = [6, 2, 1]

    g.edgeNodes = [e, h]
    g.edgeCosts = [3, 8]
    h.edgeNodes = [f, g, i]
    h.edgeCosts = [1, 8, 4]

    i.edgeNodes = [h]
    i.edgeCosts = [4]

    # build graph
    graph = [a, b, c, d, e, f, g, h, i]

    dij = Dijkstra(graph)
    path = dij.search(a, i)
    path.reverse()
    print(f"> path: {path}")
