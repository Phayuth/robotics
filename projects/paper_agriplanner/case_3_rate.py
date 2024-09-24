import os
import sys

sys.path.append(str(os.path.abspath(os.getcwd())))

import numpy as np
import matplotlib.pyplot as plt
import pickle
import pandas as pd
import pickle

from case_1_costgraph import load_cost_graph_pkl

# t = np.linspace(0, 100, 1000)
# c = 5 * 1 / np.exp(t)
# c[500:] = c[500:] - 3.0


def find_iteration_end(data):
    iterationDiffConstant = 200
    costDiffConstant = 0.01
    cBestPrevious = np.inf
    cBestPreviousIteration = 0
    for i in range(data.shape[0]):
        cNow = data[i]
        iterationNow = i
        if cNow < cBestPrevious:
            cBestPrevious = cNow
            cBestPreviousIteration = iterationNow

        if (cBestPrevious - cNow < costDiffConstant) and (iterationNow - cBestPreviousIteration > iterationDiffConstant):
            break
    return iterationNow


def plot_(data):
    plt.plot(list(data.index), data)
    # plt.plot(t[i], c[i], "*")
    plt.show()


path = "/home/yuth/ws_yuthdev/robotics_manipulator/datasave/new_data/"

i = path + "env_UR5eArmCoppeliaSimAPI_types_Multi_planner_12_withlocgap_False_costgraph.pkl"
j = path + "env_UR5eArmCoppeliaSimAPI_types_Multi_planner_16_withlocgap_False_costgraph.pkl"
k = path + "env_UR5eArmCoppeliaSimAPI_types_Multi_planner_13_withlocgap_False_costgraph.pkl"

l = path + "env_UR5eArmCoppeliaSimAPI_types_Multi_planner_12_withlocgap_True_costgraph.pkl"
m = path + "env_UR5eArmCoppeliaSimAPI_types_Multi_planner_16_withlocgap_True_costgraph.pkl"
n = path + "env_UR5eArmCoppeliaSimAPI_types_Multi_planner_13_withlocgap_True_costgraph.pkl"


def cal_rate(path):
    loadedList = load_cost_graph_pkl(path)

    arrs = [np.array(list) for list in loadedList]
    dfs = [pd.DataFrame(ar[:, 1], index=ar[:, 0].astype(np.int32), columns=[f"cost-{i}th"]) for i, ar in enumerate(arrs)]

    newIdx = pd.Index(range(2000))
    newDfs = [d.reindex(newIdx) for d in dfs]

    mergedf = pd.concat(newDfs, axis=1)
    mergedf = mergedf.interpolate(method="values", limit_direction="both")

    q = [find_iteration_end(mergedf[col]) for col in mergedf]

    qmean = np.mean(q)
    qstd = np.std(q)
    return qmean, qstd


q1m, q1s = cal_rate(i)
print(f"q1 : {q1m, q1s}")
q2m, q2s = cal_rate(j)
print(f"q2 : {q2m, q2s}")
q3m, q3s = cal_rate(k)
print(f"q3 : {q3m, q3s}")

q4m, q4s = cal_rate(l)
print(f"q4 : {q4m, q4s}")
q5m, q5s = cal_rate(m)
print(f"q5 : {q5m, q5s}")
q6m, q6s = cal_rate(n)
print(f"q6 : {q6m, q6s}")
