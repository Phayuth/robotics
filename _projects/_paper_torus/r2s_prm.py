import networkx as nx
import numpy as np
import os
from spatial_geometry.utils import Utils
from r2s_prm_plot import PlotterConfig
import matplotlib.pyplot as plt

np.set_printoptions(linewidth=2000, precision=2, suppress=True)
from pyomo.environ import (
    ConcreteModel,
    Var,
    Objective,
    SolverFactory,
    NonNegativeReals,
    Integers,
    minimize,
    Binary,
    ConstraintList,
)
import re


def solve_graph_mip(graph, startnode, endnode):
    # this looks exactly like network flow problem solved using LP
    model = ConcreteModel()

    A = []
    w = []
    for i in range(len(graph)):
        for j in range(len(graph)):
            # add any graph edge that is not 0 and tail is not startnode and head is not endnode
            if graph[i][j] != 0 and j != startnode and i != endnode:
                A.append(f"{i}x{j}")
                w.append(graph[i][j])

    model.x = Var(A, within=Binary)
    model.obj = Objective(
        expr=sum(w[i] * model.x[e] for i, e in enumerate(A)), sense=minimize
    )
    model.constraints = ConstraintList()

    print(A)
    print(w)

    startpattern = rf"^x_{startnode}.*"
    endpattern = rf"^x.*_{endnode}$"

    startconstraints = [s for s in A if re.match(startpattern, s)]
    print(f"> startconstraints: {startconstraints}")
    endconstraints = [s for s in A if re.match(endpattern, s)]
    print(f"> endconstraints: {endconstraints}")

    # start and end constraints
    # sum of start constraints must be 1
    # sum of end constraints must be 1
    model.constraints.add(expr=sum(model.x[xi] for xi in startconstraints) == 1)
    model.constraints.add(expr=sum(model.x[xi] for xi in endconstraints) == 1)

    # define the constraints for bidirectional edges
    for i in range(len(graph)):
        if i == startnode or i == endnode:
            continue
        sp = rf"^x_{i}.*"
        ep = rf"^x.*_{i}$"
        sc = [s for s in A if re.match(sp, s)]
        print(f"> sc: {sc}")
        ec = [s for s in A if re.match(ep, s)]
        print(f"> ec: {ec}")
        if len(sc) == 0 or len(ec) == 0:
            continue
        model.constraints.add(
            expr=sum(model.x[xi] for xi in sc) == sum(model.x[xi] for xi in ec)
        )
        print("add constraint")

    opt = SolverFactory("glpk")
    result_obj = opt.solve(model, tee=True)

    model.pprint()

    opt_solution = [model.x[item].value for item in A]
    print(f"> opt_solution: {opt_solution}")

    xsol = [e for i, e in enumerate(A) if opt_solution[i] == 1]
    print(f"> xsol: {xsol}")

    objective_obj = model.obj()
    print(f"> objective_obj: {objective_obj}")


class SequentialPRM:

    def __init__(self):
        self.rsrc = os.environ["RSRC_DIR"]
        self.graphmlpath = os.path.join(
            self.rsrc, "rnd_torus", "saved_prmstar_planner.graphml"
        )
        self.graphmlmatrixpath = os.path.join(
            self.rsrc, "rnd_torus", "prmstar_euclidean_matrix.npy"
        )
        self.collisionpointpath = os.path.join(
            self.rsrc, "rnd_torus", "collisionpoint_exts.npy"
        )

        self.graphml = nx.read_graphml(self.graphmlpath)
        self.colp = np.load(self.collisionpointpath)
        self.coords = self.euclidean_matrix(self.graphmlmatrixpath)

        # this not actually euclidean data, it is adjacency matrix
        # self.graphmlmatrix = nx.to_numpy_array(self.graphml)

    def euclidean_matrix(self, graphmlmatrixpath):
        if os.path.exists(graphmlmatrixpath):
            return np.load(graphmlmatrixpath)
        coords = np.array(
            [
                list(map(float, self.graphml.nodes[n]["coords"].split(",")))
                for n in self.graphml.nodes
            ]
        )
        np.save(graphmlmatrixpath, coords)
        return coords

    def nearest_node(self, query):
        dist = np.linalg.norm(self.coords - query, axis=1)
        min_idx = np.argmin(dist)
        return min_idx, f"n{min_idx}", self.graphml.nodes[f"n{min_idx}"]

    def euclidean_path_cost(self, path):
        # path lenght cost
        diffs = np.diff(path, axis=0)
        distances = np.linalg.norm(diffs, axis=1)
        return np.sum(distances)

    def get_xy(self, node):
        coords = self.graphml.nodes[node]["coords"].split(",")
        return float(coords[0]), float(coords[1])

    def query_path(self, start_np, end_np):
        startindx, start, _ = self.nearest_node(start_np)
        endindx, end, _ = self.nearest_node(end_np)

        try:
            path = nx.shortest_path(self.graphml, source=start, target=end)
            path_coords = [self.get_xy(node) for node in path]
            path_coords.insert(0, (start_np[0], start_np[1]))
            path_coords.append((end_np[0], end_np[1]))
            return np.array(path_coords), self.euclidean_path_cost(path_coords)

        except nx.NetworkXNoPath:
            print(f"no path between {start} and {end}")
            return None, np.inf

    def task_seq_matter_graph(self):
        qinit = np.array([0.40, 5.95])
        qtask1 = np.array([-3.26, 5.40])
        qtask2 = np.array([-0.03, 1.08])
        qtask3 = np.array([-2.69, -3.74])

        limt2 = np.array([[-2 * np.pi, 2 * np.pi], [-2 * np.pi, 2 * np.pi]])

        Qtask1 = Utils.find_alt_config(qtask1.reshape(2, 1), limt2).T
        Qtask2 = Utils.find_alt_config(qtask2.reshape(2, 1), limt2).T
        Qtask3 = Utils.find_alt_config(qtask3.reshape(2, 1), limt2).T

        task1numcand = Qtask1.shape[0]
        print(f"task1numcand: {task1numcand}")
        task2numcand = Qtask2.shape[0]
        print(f"task2numcand: {task2numcand}")
        task3numcand = Qtask3.shape[0]

        QQ = np.vstack((qinit, Qtask1, Qtask2, Qtask3))

        task_graph = np.zeros((QQ.shape[0], QQ.shape[0]))

        qinitvector = np.zeros((1, QQ.shape[0]))
        qinitvector[0, 0] = 0
        qinitvector[0, 1 : task1numcand + 1] = 1
        task_graph[0] = qinitvector

        idt1 = 1
        idt2 = 2
        for i in range(idt1, idt1 + task1numcand):
            for j in range(
                task1numcand - 1 + idt2, task1numcand - 1 + idt2 + task2numcand
            ):
                print(f"i: {i}, j: {j}")
                task_graph[i, j] = 1

        idt3 = 3
        for i in range(1 + task2numcand, 1 + task2numcand + task3numcand):
            for j in range(
                1 + task1numcand + task2numcand,
                1 + task1numcand + task2numcand + task3numcand,
            ):
                print(f"i: {i}, j: {j}")
                task_graph[i, j] = 1

        print(f"task_graph:")
        print(task_graph)

        task_graph_dist = np.zeros((task_graph.shape[0], task_graph.shape[0]))
        for i in range(task_graph.shape[0]):
            for j in range(task_graph.shape[0]):
                if task_graph[i, j] != 0:
                    path, cost = self.query_path(QQ[i], QQ[j])
                    if path is not None:
                        task_graph_dist[i, j] = cost

        print(f"task_graph_dist:")
        print(task_graph_dist)

        self.plot_graph(None, QQ=QQ)

        solve_graph_mip(task_graph_dist, 0, 7)

    def plot_graph(self, path, QQ=None):
        fig, ax = plt.subplots()
        ax.set_aspect("equal")
        ax.set_xlim(-2 * np.pi, 2 * np.pi)
        ax.set_ylim(-2 * np.pi, 2 * np.pi)

        # tree
        if False:
            for u, v in self.graphml.edges:
                u = self.graphml.nodes[u]["coords"].rsplit(",")
                v = self.graphml.nodes[v]["coords"].rsplit(",")
                ax.plot(
                    [float(u[0]), float(v[0])],
                    [float(u[1]), float(v[1])],
                    color=PlotterConfig.treeColor,
                    linewidth=PlotterConfig.globalLinewidth,
                    marker=PlotterConfig.treeMarker,
                    markerfacecolor=PlotterConfig.treeFaceColor,
                    markersize=PlotterConfig.treeMarkersize,
                )
        if QQ is not None:
            ax.set_title("sequence is blue->red->yellow->green")
            for i in range(QQ.shape[0]):
                if i == 0:
                    color = PlotterConfig.stateStartColor
                if i in [1, 2, 3, 4]:
                    color = "red"
                if i in [5, 6, 7, 8]:
                    color = "yellow"
                if i in [9, 10, 11, 12]:
                    color = "green"

                ax.plot(
                    QQ[i, 0],
                    QQ[i, 1],
                    color=color,
                    linewidth=PlotterConfig.globalLinewidth,
                    marker=PlotterConfig.stateMarker,
                    markerfacecolor=color,
                    markersize=PlotterConfig.stateMarkersize,
                )

        # collision points
        ax.plot(
            self.colp[:, 0],
            self.colp[:, 1],
            color="darkcyan",
            linewidth=0,
            marker="o",
            markerfacecolor="darkcyan",
            markersize=1.5,
        )

        if path is not None:
            ax.plot(
                path[:, 0],
                path[:, 1],
                color=PlotterConfig.pathColor,
                linewidth=PlotterConfig.globalLinewidth,
                marker=PlotterConfig.pathMarker,
                markerfacecolor=PlotterConfig.pathFaceColor,
                markersize=PlotterConfig.pathMarkersize,
            )

        plt.show()


if __name__ == "__main__":
    prm = SequentialPRM()

    start = np.array([0.0, 0.0])
    end = np.array([-4, -6])
    path, cost = prm.query_path(start, end)
    print("path", path)
    print("cost", cost)

    prm.plot_graph(path, QQ=None)

    # prm.query_task()

    prm.task_seq_matter_graph()
