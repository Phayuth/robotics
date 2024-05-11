import os
import sys

wd = os.path.abspath(os.getcwd())
sys.path.append(str(wd))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker
import matplotlib
import pickle
import pandas as pd


matplotlib.rcParams.update({"font.size": 10})


def compose_path(p):
    return f"./datasave/planner_performance/single2d/{p}_costgraph.pkl"


def load_cost_graph_pkl(path):
    with open(path, "rb") as file:
        loadedList = pickle.load(file)
    return loadedList


def mean_curve(loadedList):
    arrs = [np.array(list) for list in loadedList]
    dfs = [pd.DataFrame(ar[:, 1], index=ar[:, 0].astype(np.int32), columns=[f"cost-{i}th"]) for i, ar in enumerate(arrs)]

    newIdx = pd.Index(range(2000))
    newDfs = [d.reindex(newIdx) for d in dfs]

    mergedf = pd.concat(newDfs, axis=1)
    mergedf = mergedf.interpolate(method="values", limit_direction="both")

    # an.to_csv("out.csv", index=True)
    meandfs = mergedf.mean(axis=1)

    return meandfs


def get_(p):
    path = compose_path(p)
    ll = load_cost_graph_pkl(path)
    return mean_curve(ll)


matplotlib.rcParams.update({"font.size": 10})
scale = 0.9
fig = plt.figure(figsize=(scale * 3.40067 * 2, scale * 3.40067), frameon=True, layout="tight")
ax = plt.subplot(1, 1, 1)

nameList = ["informed2d", "connect2d", "connectlocgap2d", "connectstar2d", "connectstarlocgap2d"]

meandf_inf = get_(nameList[0])
meandf_cnt = get_(nameList[1])
meandf_cntlg = get_(nameList[2])
meandf_cnts = get_(nameList[3])
meandf_cntslg = get_(nameList[4])

ax.plot(meandf_inf.index, meandf_inf.values, label="Inform-RRT*", linewidth=3, linestyle="solid", color="#558b2f")
ax.plot(meandf_cnt.index, meandf_cnt.values, label="RRT-Connect", linewidth=3, linestyle="solid", color="#ff8f00")
ax.plot(meandf_cntlg.index, meandf_cntlg.values, label="RRT-Connect + LocGap", linewidth=3, linestyle="--", color="#1565c0")
ax.plot(meandf_cnts.index, meandf_cnts.values, label="RRT*-Connect", linewidth=3, linestyle="--", color="#9e9d24")
ax.plot(meandf_cntslg.index, meandf_cntslg.values, label="RRT*-Connect + LocGap", linewidth=3, linestyle="--", color="#6a1b9a")

ax.set_xlabel("Iteration")
ax.set_ylabel("Cost")
ax.grid(axis="y")
ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
ax.tick_params(axis="both", which="major", direction="in", length=3, width=1, colors="black", grid_color="gray", grid_alpha=0.2)
ax.tick_params(axis="both", which="minor", direction="in", length=1, width=1, colors="black", grid_color="gray", grid_alpha=0.1)
ax.set_xlim((0, 2025))
ax.legend(loc=1)
plt.show()
