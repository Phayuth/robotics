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

# matplotlib.rcParams.update({"font.size": 12})
# plt.rcParams['svg.fonttype'] = 'none'

plt.rc('text', usetex=True)
plt.rc('font', family='serif')

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


def get_(path):
    ll = load_cost_graph_pkl(path)
    return mean_curve(ll)


def single():
    path = "/home/yuth/ws_yuthdev/robotics_manipulator/datasave/new_data/"

    a = path + "env_RobotArm2DSimulator_types_Single_planner_2_withlocgap_False_costgraph.pkl"
    b = path + "env_RobotArm2DSimulator_types_Single_planner_4_withlocgap_False_costgraph.pkl"
    c = path + "env_RobotArm2DSimulator_types_Single_planner_15_withlocgap_False_costgraph.pkl"
    d = path + "env_RobotArm2DSimulator_types_Single_planner_5_withlocgap_False_costgraph.pkl"
    e = path + "env_RobotArm2DSimulator_types_Single_planner_6_withlocgap_False_costgraph.pkl"

    f = path + "env_RobotArm2DSimulator_types_Single_planner_4_withlocgap_True_costgraph.pkl"
    g = path + "env_RobotArm2DSimulator_types_Single_planner_15_withlocgap_True_costgraph.pkl"
    h = path + "env_RobotArm2DSimulator_types_Single_planner_5_withlocgap_True_costgraph.pkl"

    meandf_inf = get_(a)
    meandf_cnt = get_(b)
    meandf_cntna = get_(c)
    meandf_starcnt = get_(d)
    meandf_infstarcnt = get_(e)

    meandf_cntlocgap = get_(f)
    meandf_cntnalocgap = get_(g)
    meandf_starcntlocgap = get_(h)

    scale = 1.4
    aspect = 16/9
    size_aspect = scale * 3.40067
    lw = 1
    fig = plt.figure(figsize=(size_aspect*aspect, size_aspect), frameon=True, layout="tight")
    ax = plt.subplot(1, 1, 1)

    ax.plot(meandf_inf.index, meandf_inf.values, label="Infm-RRT*", linewidth=lw, linestyle="solid", color="#558b2f")
    ax.plot(meandf_cnt.index, meandf_cnt.values, label="RRT-Cnt", linewidth=lw, linestyle="solid", color="#ff8f00")
    ax.plot(meandf_cntna.index, meandf_cntna.values, label="RRT-Cnt NA", linewidth=lw, linestyle="solid", color="#1565c0")
    ax.plot(meandf_starcnt.index, meandf_starcnt.values, label="RRT*-Cnt", linewidth=lw, linestyle="solid", color="#9e9d24")
    ax.plot(meandf_infstarcnt.index, meandf_infstarcnt.values, label="Infm-RRT*-Cnt", linewidth=lw, linestyle="solid", color="#6a1b9a")

    ax.plot(meandf_cntlocgap.index, meandf_cntlocgap.values, label="RRT-Cnt + LG", linewidth=lw, linestyle="--", color="#a218e1")
    ax.plot(meandf_cntnalocgap.index, meandf_cntnalocgap.values, label="RRT-Cnt NA + LG", linewidth=lw, linestyle="--", color="#e1182a")
    ax.plot(meandf_starcntlocgap.index, meandf_starcntlocgap.values, label="RRT*-Cnt + LG", linewidth=lw, linestyle="--", color="#e118ad")

    # 3.9831719583628873 coptimal
    ax.plot(meandf_cntlocgap.index, [3.9831719583628873]*len(meandf_cntlocgap.index), linewidth=lw, linestyle=":", color="k")

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Cost")
    ax.grid(axis="y")
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.tick_params(axis="both", which="major", direction="in", length=3, width=1, colors="black", grid_color="gray", grid_alpha=0.2)
    ax.tick_params(axis="both", which="minor", direction="in", length=1, width=1, colors="black", grid_color="gray", grid_alpha=0.1)
    ax.set_xlim((0, 2025))
    # ax.legend(loc=1)
    ax.legend(ncols=4, bbox_to_anchor=(0, 1), loc='lower left')
    plt.show()


def multi():
    path = "/home/yuth/ws_yuthdev/robotics_manipulator/datasave/new_data/"

    i = path + "env_RobotArm2DSimulator_types_Multi_planner_12_withlocgap_False_costgraph.pkl"
    j = path + "env_RobotArm2DSimulator_types_Multi_planner_16_withlocgap_False_costgraph.pkl"
    k = path + "env_RobotArm2DSimulator_types_Multi_planner_13_withlocgap_False_costgraph.pkl"

    l = path + "env_RobotArm2DSimulator_types_Multi_planner_12_withlocgap_True_costgraph.pkl"
    m = path + "env_RobotArm2DSimulator_types_Multi_planner_16_withlocgap_True_costgraph.pkl"
    n = path + "env_RobotArm2DSimulator_types_Multi_planner_13_withlocgap_True_costgraph.pkl"

    meandf_cntml = get_(i)
    meandf_cntnaml = get_(j)
    meandf_starcntml = get_(k)

    meandf_cntmllg = get_(l)
    meandf_cntnamllg = get_(m)
    meandf_starcntmllg = get_(n)

    scale = 1.13
    aspect = 16/9
    size_aspect = scale * 3.40067
    lw = 1.5
    fig = plt.figure(figsize=(size_aspect*aspect, size_aspect), frameon=True, layout="tight")
    ax = plt.subplot(1, 1, 1)

    fi_cntml = 103
    fi_cntnaml = 125
    fi_starcntml = 103

    fi_cntmllg = 103
    fi_cntnamllg = 125
    fi_starcntmllg = 103

    ax.plot(meandf_cntnaml.index[fi_cntnaml:], meandf_cntnaml.values[fi_cntnaml:], label="Bi-RRT", linewidth=lw, linestyle="solid", color="#ff8f00")
    ax.plot(meandf_cntnamllg.index[fi_cntnamllg:], meandf_cntnamllg.values[fi_cntnamllg:], label="Bi-RRT + LocGap", linewidth=lw, linestyle="--", color="#ff8f00")

    ax.plot(meandf_cntml.index[fi_cntml:], meandf_cntml.values[fi_cntml:], label="RRT-Cnt", linewidth=lw, linestyle="solid", color="#558b2f")
    ax.plot(meandf_cntmllg.index[fi_cntmllg:], meandf_cntmllg.values[fi_cntmllg:], label="RRT-Cnt + LocGap", linewidth=lw, linestyle="--", color="#558b2f")

    ax.plot(meandf_starcntml.index[fi_starcntml:], meandf_starcntml.values[fi_starcntml:], label="RRT*-Cnt", linewidth=lw, linestyle="solid", color="#1565c0")
    ax.plot(meandf_starcntmllg.index[fi_starcntmllg:], meandf_starcntmllg.values[fi_starcntmllg:], label="RRT*-Cnt + LocGap", linewidth=lw, linestyle="--", color="#1565c0")

    # 3.8304068904535753 coptimal
    ax.plot(meandf_cntml.index, [3.8304068904535753]*len(meandf_cntml.index), linewidth=lw, linestyle=":", color="k")

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Cost")
    ax.grid(axis="y")
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.tick_params(axis="both", which="major", direction="in", length=5, width=1, colors="black", grid_color="gray", grid_alpha=0.2)
    ax.tick_params(axis="both", which="minor", direction="in", length=3, width=1, colors="black", grid_color="gray", grid_alpha=0.1)
    ax.set_xlim((0, 2025))
    # ax.legend(loc=1)
    ax.legend(ncols=3, bbox_to_anchor=(0, 1), loc='lower left')
    # ax.legend()
    plt.show()


if __name__=="__main__":
    # single()
    multi()