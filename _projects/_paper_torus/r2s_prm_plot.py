import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import os


class PlotterConfig:
    globalLinewidth = 1

    obstColor = "darkcyan"
    obstFaceColor = "darkcyan"
    obstMarker = "o"
    obstMarkersize = 1.5

    treeColor = "darkgray"
    treeFaceColor = None
    treeMarker = None
    treeMarkersize = None

    stateStartColor = "blue"
    stateStartFaceColor = "yellow"
    stateAppColor = "blue"
    stateAppFaceColor = "green"
    stateGoalColor = "blue"
    stateGoalFaceColor = "red"
    stateMarkersize = 7
    stateMarker = "o"

    pathColor = "blue"
    pathFaceColor = "plum"
    pathMarker = "o"
    pathMarkersize = 7


rsrc = os.environ["RSRC_DIR"]
graph = nx.read_graphml(os.path.join(rsrc, "rnd_torus", "saved_planner.graphml"))
colp = np.load(os.path.join(rsrc, "rnd_torus", "collisionpoint_exts.npy"))

fig, ax = plt.subplots()
ax.set_aspect("equal")
ax.set_xlim(-2 * np.pi, 2 * np.pi)
ax.set_ylim(-2 * np.pi, 2 * np.pi)
# tree
for u, v in graph.edges:
    u = graph.nodes[u]["coords"].rsplit(",")
    v = graph.nodes[v]["coords"].rsplit(",")
    ax.plot(
        [float(u[0]), float(v[0])],
        [float(u[1]), float(v[1])],
        color=PlotterConfig.treeColor,
        linewidth=PlotterConfig.globalLinewidth,
        marker=PlotterConfig.treeMarker,
        markerfacecolor=PlotterConfig.treeFaceColor,
        markersize=PlotterConfig.treeMarkersize,
    )

# collision points
ax.plot(
    colp[:, 0],
    colp[:, 1],
    color="darkcyan",
    linewidth=0,
    marker="o",
    markerfacecolor="darkcyan",
    markersize=1.5,
)

# for i in range(600):
#     for j in range(600):
#         s = f"n{i}"
#         t = f"n{j}"
#         try:
#             path = nx.shortest_path(graph, source=s, target=t)
#             if len(path) > 8:
#                 print(path)
#         except nx.NetworkXNoPath:
#             print(f"no path between {s} and {t}")
#             continue

path = nx.shortest_path(graph, source="n309", target="n218")
s = graph.nodes["n309"]["coords"].rsplit(",")
t = graph.nodes["n218"]["coords"].rsplit(",")
path = [graph.nodes[p]["coords"].rsplit(",") for p in path]


ax.plot(
    [float(p[0]) for p in path],
    [float(p[1]) for p in path],
    color=PlotterConfig.pathColor,
    linewidth=PlotterConfig.globalLinewidth,
    marker=PlotterConfig.pathMarker,
    markerfacecolor=PlotterConfig.pathFaceColor,
    markersize=PlotterConfig.pathMarkersize,
)

plt.show()
