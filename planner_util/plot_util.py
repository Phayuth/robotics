import os
import sys

wd = os.path.abspath(os.getcwd())
sys.path.append(str(wd))

import numpy as np
import matplotlib.pyplot as plt
from planner_util.extract_path_class import extract_path_class_6d


def plot_tree(treeVertex, path):
    for vertex in treeVertex:
        plt.scatter(vertex.x, vertex.y, color="sandybrown")
        if vertex.parent == None:
            pass
        else:
            plt.plot([vertex.x, vertex.parent.x], [vertex.y, vertex.parent.y], color="green")

    if path:
        plt.plot([node.x for node in path], [node.y for node in path], color='blue')


def plot_tree_3d(treeVertex, path):
    ax = plt.axes(projection='3d')

    for vertex in treeVertex:
        # ax.scatter(vertex.x, vertex.y, vertex.z, color="sandybrown")
        if vertex.parent == None:
            pass
        else:
            ax.plot3D([vertex.x, vertex.parent.x], [vertex.y, vertex.parent.y], [vertex.z, vertex.parent.z], "green")

    if path:
        ax.plot3D([node.x for node in path], [node.y for node in path], [node.x for node in path], color='blue')


def plot_joint_6d(path, timeEnd, timeStep):

    time = np.linspace(0, timeEnd, timeStep)
    pathX, pathY, pathZ, pathP, pathQ, pathR = extract_path_class_6d(path)

    fig, axes = plt.subplots(6, 1)
    axes[0].plot(time, pathX)
    axes[1].plot(time, pathY)
    axes[2].plot(time, pathZ)
    axes[3].plot(time, pathP)
    axes[4].plot(time, pathQ)
    axes[5].plot(time, pathR)
