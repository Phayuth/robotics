import numpy as np
import json


class Node:

    def __init__(self, config=None, name=None) -> None:
        self.name = name
        self.config = config
        self.cost = np.inf
        self.pathvia: Node = None
        self.edgeNodes: list[Node] = []
        self.edgeCosts: list[float] = []

    def __repr__(self) -> str:
        return f"n:cf{self.config.T}"


def save_graph(graph, path):
    data = []
    for gc in graph:
        n = {
            "config": gc.config.flatten().tolist(),
            "name": gc.name,
            "edgeNodesID": [graph.index(gi) for gi in gc.edgeNodes if len(gc.edgeNodes) != 0],
            "edgeCosts": [gi for gi in gc.edgeCosts if len(gc.edgeCosts) != 0],
        }
        data.append(n)

    with open(f"{path}.json", "w") as file:
        json.dump(data, file, indent=4)


def load_graph(path):
    with open(f"{path}.json", "r") as file:
        dictloaded = json.load(file)

    numnode = len(dictloaded)
    g = [Node() for _ in range(numnode)]
    for i in range(numnode):
        g[i].config = np.array(dictloaded[i]["config"]).reshape(-1, 1)
        g[i].name = dictloaded[i]["name"]
        if len(dictloaded[i]["edgeNodesID"]) != 0:
            for nid in dictloaded[i]["edgeNodesID"]:
                g[i].edgeNodes.append(g[nid])
            g[i].edgeCosts = dictloaded[i]["edgeCosts"]

    return g


if __name__ == "__main__":

    import os
    import sys

    sys.path.append(str(os.path.abspath(os.getcwd())))

    import matplotlib.pyplot as plt
    import numpy as np

    from planner.graph_based.dijkstra import Node, Dijkstra

    # create node
    a = Node(config=np.random.uniform(low=0, high=1, size=(2, 1)), name="a")
    b = Node(config=np.random.uniform(low=0, high=1, size=(2, 1)), name="b")
    c = Node(config=np.random.uniform(low=0, high=1, size=(2, 1)), name="c")
    d = Node(config=np.random.uniform(low=0, high=1, size=(2, 1)), name="d")
    e = Node(config=np.random.uniform(low=0, high=1, size=(2, 1)), name="e")
    f = Node(config=np.random.uniform(low=0, high=1, size=(2, 1)), name="f")
    g = Node(config=np.random.uniform(low=0, high=1, size=(2, 1)), name="g")
    h = Node(config=np.random.uniform(low=0, high=1, size=(2, 1)), name="h")
    i = Node(config=np.random.uniform(low=0, high=1, size=(2, 1)), name="i")

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

    save_graph(graph)
    g = load_graph()
    print(f"> g: {g}")

    # search single goal
    dij = Dijkstra(g)
    path = dij.search(g[0], g[-1])
    path.reverse()
    print(f"> path: {path}")
