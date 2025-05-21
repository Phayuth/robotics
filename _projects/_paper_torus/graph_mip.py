import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

np.random.seed(9)
np.set_printoptions(precision=2, suppress=True, linewidth=200)


def solve_graph_mip(graph, startnode=0, endnode=5):
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

    # this looks exactly like network flow problem solved using LP
    model = ConcreteModel()

    A = []
    w = []
    for i in range(len(graph)):
        for j in range(len(graph)):
            # add any graph edge that is not 0 and tail is not startnode and head is not endnode
            if graph[i][j] != 0 and j != startnode and i != endnode:
                A.append(f"{i}_x_{j}")
                w.append(graph[i][j])

    model.x = Var(A, within=Binary)
    model.obj = Objective(
        expr=sum(w[i] * model.x[e] for i, e in enumerate(A)), sense=minimize
    )
    model.constraints = ConstraintList()

    print(A)
    print(w)

    startpattern = rf"{startnode}_x_.*"
    endpattern = rf"^.*_x_{endnode}$"

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
        sp = rf"{i}_x_.*"
        ep = rf"^.*_x_{i}$"
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
    xsolv = [i for i, e in enumerate(A) if opt_solution[i] == 1]
    print(f"> xsol: {xsol}")
    print(f"> xsolv: {xsolv}")

    objective_obj = model.obj()
    print(f"> objective_obj: {objective_obj}")

    result = []
    for item in xsol:
        a, b = map(int, item.split("_x_"))
        if a not in result:
            result.append(a)
        if b not in result:
            result.append(b)
    print(f"> result: {result}")
    return result


def multi_target_dijkstra(adj_matrix, source, targets):
    G = nx.DiGraph()
    rows, cols = np.nonzero(adj_matrix)

    for i, j in zip(rows, cols):
        G.add_edge(i, j, weight=adj_matrix[i, j])

    # Run Dijkstra and get full shortest paths
    lengths, paths = nx.single_source_dijkstra(G, source)

    # Get closest target
    closest_target = min(targets, key=lambda t: lengths.get(t, float("inf")))

    print("Closest target:", closest_target)
    print("Path:", paths[closest_target])
    print("Cost:", lengths[closest_target])

    return [int(p) for p in paths[closest_target]]


if __name__ == "__main__":
    # graph = np.array(
    #     [
    #         [0, 35, 30, 20, 0, 0, 0],
    #         [35, 0, 8, 0, 12, 0, 0],
    #         [30, 8, 0, 9, 10, 20, 0],
    #         [20, 0, 9, 0, 0, 0, 15],
    #         [0, 12, 10, 0, 0, 5, 20],
    #         [0, 0, 20, 0, 5, 0, 5],
    #         [0, 0, 0, 15, 20, 5, 0],
    #     ]
    # )
    # solve_graph_mip(graph)
    # visualize_bidirectional_graph(graph)
    # import re

    # list = [f"{i}_x_{j}" for i in range(50) for j in range(50)]
    # print(list)

    # startpattern = rf"5_x_.*"
    # endpattern = rf"^.*_x_15$"

    # print("-" * 20)
    # a = [s for s in list if re.match(startpattern, s)]
    # print(f"> a: {a}")

    # print("-" * 20)
    # b = [s for s in list if re.match(endpattern, s)]
    # print(f"> b: {b}")

    # print("-" * 20)

    # task_candidates_num = np.array([0, 1, 4, 4, 4])
    # node_num = task_candidates_num.sum()
    # cs = np.cumsum(task_candidates_num)
    # numtask = task_candidates_num.shape[0] - 2

    # print(f"numtask: {numtask}")
    # print(f"node_num: {node_num}")
    # print(f"task_candidates_num: {task_candidates_num}")
    # print(f"cs: {cs}")

    # adj_matrix = np.zeros((node_num, node_num))
    # print(f"adj_matrix:")
    # print(adj_matrix)
    # # k = 2
    # # i = range(cs[k], cs[k + 1])
    # # j = range(cs[k + 1], cs[k + 2])
    # # print(f"i: {i}")
    # # for iii in i:
    # #     print(f"iii: {iii}")
    # # print(f"j: {j}")
    # # for jjj in j:
    # #     print(f"jjj: {jjj}")

    # for k in range(numtask):
    #     for i in range(cs[k], cs[k + 1]):
    #         for j in range(cs[k + 1], cs[k + 2]):
    #             adj_matrix[i, j] = 1

    # print(f"adj_matrix:")
    # print(adj_matrix)

    task_candidates_num = np.array([0, 1, 4, 4, 4])
    node_num = task_candidates_num.sum()
    cs = np.cumsum(task_candidates_num)
    numtask = task_candidates_num.shape[0] - 2

    # id = 2
    # i = range(0, node_num)
    # j = range(0, node_num)
    # ii = list(range(cs[id], cs[id + 1]))
    # jj = list(range(cs[id], cs[id + 1]))
    # print(f"i: {ii}")
    # print(f"j: {jj}")

    adj_matrix = np.ones((node_num, node_num))
    print(f"adj_matrix:")
    print(adj_matrix)

    for k in range(numtask + 1):
        for i in range(cs[k], cs[k + 1]):
            for j in range(cs[k], cs[k + 1]):
                adj_matrix[i, j] = 0

    print(f"adj_matrix:")
    print(adj_matrix)
