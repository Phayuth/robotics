import os
import sys

sys.path.append(str(os.path.abspath(os.getcwd())))

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN
from scipy.spatial import ConvexHull
from e2d_concavehull import ConcaveHull
from simulator.sim_planar_rr import RobotArm2DSimulator
from e2d_parameters import PaperLengths

plt.rcParams["svg.fonttype"] = "none"


def naive_plot():
    env = RobotArm2DSimulator(torusspace=False)

    wd = 6.68459 * 0.32
    ht = 6.68459 * 0.32
    fig = plt.figure(figsize=(wd, ht), frameon=True, layout="tight", dpi=600)
    ax = plt.subplot(1, 1, 1)
    ax.set_aspect("equal")
    env.plot_cspace(ax)
    # ax.set_xlabel("$\\theta_1$")
    # ax.set_ylabel("$\\theta_2$")
    ax.set_xticklabels(["$-\\pi$", "$\\theta_1$", "$\\pi$"])
    ax.set_yticklabels(["$-\\pi$", "$\\theta_2$", "$\\pi$"])
    ax.set_xticks([-np.pi, 0, np.pi])
    ax.set_yticks([-np.pi, 0, np.pi])
    ax.set_xlim(-np.pi, np.pi)
    ax.set_ylim(-np.pi, np.pi)
    # ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    # ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    # ax.tick_params(axis="both", which="major", direction="in", length=5, width=1, colors="black", grid_color="gray", grid_alpha=0.2)
    # ax.tick_params(axis="both", which="minor", direction="in", length=3, width=1, colors="black", grid_color="gray", grid_alpha=0.1)
    plt.savefig("/home/yuth/exp1_cspace.png", bbox_inches="tight")


def better_plot():
    xy_points = np.load("./datasave/planner_ijcas_data/collisionpoint_so2s.npy")

    num_clusters = 5
    kmeans = DBSCAN()
    cluster_labels = kmeans.fit_predict(xy_points)
    cluster_lists = [[] for _ in range(num_clusters)]
    for i, label in enumerate(cluster_labels):
        cluster_lists[label].append(xy_points[i])

    wd = 0.30 * PaperLengths.latex_textwidth_inch
    ht = 0.30 * PaperLengths.latex_textwidth_inch
    fig = plt.figure(figsize=(wd, ht), frameon=True, layout="tight", dpi=600)
    ax = plt.subplot(1, 1, 1)
    ax.set_aspect("equal")

    # colors = ["red", "green", "blue", "orange", "purple"]
    colors = ["#2ca08980"] * num_clusters
    for i, cluster in enumerate(cluster_lists):
        cluster = np.array(cluster)

        # convex patch
        # hull = ConvexHull(cluster)
        # boundary_points = cluster[hull.vertices]
        # cluster_polygon = plt.Polygon(boundary_points, edgecolor=colors[i], facecolor="none")
        # ax.add_patch(cluster_polygon)
        # plt.scatter(cluster[:, 0], cluster[:, 1], color=colors[i], label=f"Cluster {i + 1}")

        # concave patch
        hull = ConcaveHull(cluster, 5)
        boundary_points = hull.calculate()
        # cluster_polygon = plt.Polygon(boundary_points, edgecolor=colors[i], facecolor=colors[i], alpha=0.5)
        cluster_polygon = plt.Polygon(boundary_points, edgecolor="k", facecolor=colors[i], linewidth=0.2 * PaperLengths.mmtopoint)
        ax.add_patch(cluster_polygon)

    for axis in ["top", "bottom", "left", "right"]:
        ax.spines[axis].set_linewidth(1)
    ax.set_xticklabels(["$-\pi\$", "$\\theta_1\$", "$\pi\$"])
    ax.set_yticklabels(["$-\pi\$", "$\\theta_2\$", "$\pi\$"])
    ax.set_xticks([-2.8, 0, 2.8])
    ax.set_yticks([-2.8, 0, 2.8])
    ax.set_xlim(-np.pi, np.pi)
    ax.set_ylim(-np.pi, np.pi)
    ax.tick_params(axis="both", which="major", direction="in", length=0, width=1, colors="black", grid_color="gray", grid_alpha=0.2, labelsize=0.1)
    ax.tick_params(axis="both", which="minor", direction="in", length=0, width=1, colors="black", grid_color="gray", grid_alpha=0.1, labelsize=0.1)
    # plt.savefig("/home/yuth/exp1_cspace.png", bbox_inches="tight")
    plt.savefig("/home/yuth/exp1_cspace.svg", bbox_inches="tight", format="svg", transparent=True)


# better_plot()


def plot_cobs_concavehull(ax):
    xy_points = np.load("./datasave/planner_ijcas_data/collisionpoint_so2s.npy")

    num_clusters = 5
    kmeans = DBSCAN()
    cluster_labels = kmeans.fit_predict(xy_points)
    cluster_lists = [[] for _ in range(num_clusters)]
    for i, label in enumerate(cluster_labels):
        cluster_lists[label].append(xy_points[i])

    colors = ["#2ca08980"] * num_clusters
    for i, cluster in enumerate(cluster_lists):
        cluster = np.array(cluster)

        # concave patch
        hull = ConcaveHull(cluster, 5)
        boundary_points = hull.calculate()
        # cluster_polygon = plt.Polygon(boundary_points, edgecolor=colors[i], facecolor=colors[i], alpha=0.5)
        cluster_polygon = plt.Polygon(boundary_points, edgecolor="k", facecolor=colors[i], linewidth=0.2 * PaperLengths.mmtopoint)
        ax.add_patch(cluster_polygon)
