import os
import sys

sys.path.append(str(os.path.abspath(os.getcwd())))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker
import matplotlib
import pickle
import pandas as pd

# matplotlib.rcParams.update({"font.size": 10})
# plt.rcParams['svg.fonttype'] = 'none'

# def load_cost_graph_pkl(path):
#     with open(path, "rb") as file:
#         loadedList = pickle.load(file)
#     return loadedList

def load_cost_graph_pkl(path):
    with open(path, "rb") as file:
        loadedList = pickle.load(file)

    countfail = np.sum([1 for innerlist in loadedList if len(innerlist) != 0])
    print(countfail)

    filterlist = []
    for i in range(len(loadedList)):
        if len(loadedList[i]) != 0:
            filterlist.append(loadedList[i])

    return filterlist


def mean_curve(loadedList):
    arrs = [np.array(list) for list in loadedList]
    dfs = [pd.DataFrame(ar[:, 1], index=ar[:, 0].astype(np.int32), columns=[f"cost-{i}th"]) for i, ar in enumerate(arrs)]

    newIdx = pd.Index(range(3000))
    newDfs = [d.reindex(newIdx) for d in dfs]

    mergedf = pd.concat(newDfs, axis=1)
    mergedf = mergedf.interpolate(method="values", limit_direction="both")

    # an.to_csv("out.csv", index=True)
    meandfs = mergedf.mean(axis=1)
    return meandfs


def get_(path):
    ll = load_cost_graph_pkl(path)
    return mean_curve(ll)


def multi():
    path = "/home/yuth/ws_yuthdev/robotics_manipulator/datasave/new_data/"

    i = path + "env_UR5eArmCoppeliaSimAPI_types_Multi_planner_12_withlocgap_False_costgraph.pkl"
    j = path + "env_UR5eArmCoppeliaSimAPI_types_Multi_planner_16_withlocgap_False_costgraph.pkl"
    k = path + "env_UR5eArmCoppeliaSimAPI_types_Multi_planner_13_withlocgap_False_costgraph.pkl"

    l = path + "env_UR5eArmCoppeliaSimAPI_types_Multi_planner_12_withlocgap_True_costgraph.pkl"
    m = path + "env_UR5eArmCoppeliaSimAPI_types_Multi_planner_16_withlocgap_True_costgraph.pkl"
    n = path + "env_UR5eArmCoppeliaSimAPI_types_Multi_planner_13_withlocgap_True_costgraph.pkl"

    meandf_cntml = get_(i)
    meandf_cntnaml = get_(j)
    meandf_starcntml = get_(k)

    meandf_cntmllg = get_(l)
    meandf_cntnamllg = get_(m)
    meandf_starcntmllg = get_(n)

    # fig setup
    wd = 3.19423  # inches
    ht = 3.19423 / 2 + 0.5  # inches
    lw = 1
    fontsz = 6
    fig = plt.figure(figsize=(wd, ht), frameon=True, layout="tight", dpi=600)
    ax = plt.subplot(1, 1, 1)
    ax.set_xlabel("Iteration", fontsize=fontsz)
    ax.set_ylabel("Cost", fontsize=fontsz)
    ax.grid(axis="y")
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.tick_params(axis="both", which="major", direction="in", length=5, width=1, colors="black", grid_color="gray", grid_alpha=0.2, labelsize=fontsz)
    ax.tick_params(axis="both", which="minor", direction="in", length=3, width=1, colors="black", grid_color="gray", grid_alpha=0.1, labelsize=fontsz)
    ax.set_xlim((0, 2025))

    fi_cntml = 168
    fi_cntnaml = 240
    fi_starcntml = 168

    fi_cntmllg = 168
    fi_cntnamllg = 240
    fi_starcntmllg = 168

    ax.plot(meandf_cntnaml.index[fi_cntnaml:], meandf_cntnaml.values[fi_cntnaml:], label="Bi-RRT", linewidth=lw, linestyle="solid", color="#ff8f00")
    ax.plot(meandf_cntnamllg.index[fi_cntnamllg:], meandf_cntnamllg.values[fi_cntnamllg:], label="Bi-RRT & LG", linewidth=lw, linestyle="--", color="#ff8f00")

    ax.plot(meandf_cntml.index[fi_cntml:], meandf_cntml.values[fi_cntml:], label="RRT-Cnt", linewidth=lw, linestyle="solid", color="#558b2f")
    ax.plot(meandf_cntmllg.index[fi_cntmllg:], meandf_cntmllg.values[fi_cntmllg:], label="RRT-Cnt & LG", linewidth=lw, linestyle="--", color="#558b2f")

    ax.plot(meandf_starcntml.index[fi_starcntml:], meandf_starcntml.values[fi_starcntml:], label="RRT*-Cnt", linewidth=lw, linestyle="solid", color="#1565c0")
    ax.plot(meandf_starcntmllg.index[fi_starcntmllg:], meandf_starcntmllg.values[fi_starcntmllg:], label="RRT*-Cnt & LG", linewidth=lw, linestyle="--", color="#1565c0")

    # 3.8304068904535753 coptimal
    # ax.plot(meandf_cntml.index, [3.8304068904535753]*len(meandf_cntml.index), linewidth=lw, linestyle=":", color="k")

    ax.legend(bbox_to_anchor=(0.0, 1.02, 1.0, 0.102), loc="lower left", ncols=3, mode="expand", borderaxespad=0.0, edgecolor="k", fancybox=False, fontsize=fontsz)
    plt.savefig("/home/yuth/exp3_costplot.svg", bbox_inches="tight")
    # plt.savefig("/home/yuth/exp3_costplot.png", bbox_inches="tight")
    # plt.savefig("/home/yuth/exp3_costplot.pdf", bbox_inches="tight")
    # plt.show()


if __name__=="__main__":
    multi()