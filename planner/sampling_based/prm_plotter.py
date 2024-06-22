import os
import sys

wd = os.path.abspath(os.getcwd())
sys.path.append(str(wd))

import numpy as np
import matplotlib.patches as mpatches


class PRMPlotter:

    def plot_2d_obstacle(simulator, axis):
        joint1Range = np.linspace(simulator.configLimit[0][0], simulator.configLimit[0][1], 360)
        joint2Range = np.linspace(simulator.configLimit[1][0], simulator.configLimit[1][1], 360)
        collisionPoint = []
        for theta1 in joint1Range:
            for theta2 in joint2Range:
                config = np.array([[theta1], [theta2]])
                result = simulator.collision_check(config)
                if result is True:
                    collisionPoint.append([theta1, theta2])

        collisionPoint = np.array(collisionPoint)
        axis.plot(collisionPoint[:, 0], collisionPoint[:, 1], color="darkcyan", linewidth=0, marker="o", markerfacecolor="darkcyan", markersize=1.5)

    # def plot_2d_roadmap(edges, axis):
    #     for edge in edges:
    #         axis.plot([edge.nodeA.config[0], edge.nodeB.config[0]], [edge.nodeA.config[1], edge.nodeB.config[1]], color="darkgray")

    def plot_2d_roadmap(nodes, axis):
        for ns in nodes:
            for nc in ns.edgeNodes:
                axis.plot([ns.config[0], nc.config[0]], [ns.config[1], nc.config[1]], color="darkgray")

    def plot_2d_state_configuration(xStart, xGoal, axis):
        axis.plot(xStart.config[0], xStart.config[1], color="blue", linewidth=0, marker="o", markerfacecolor="yellow")

        if isinstance(xGoal, list):
            for xG in xGoal:
                axis.plot(xG.config[0], xG.config[1], color="blue", linewidth=0, marker="o", markerfacecolor="red")
        else:
            axis.plot(xGoal.config[0], xGoal.config[1], color="blue", linewidth=0, marker="o", markerfacecolor="red")

    def plot_2d_path(path, axis):
        axis.plot([p.config[0] for p in path], [p.config[1] for p in path], color="blue", linewidth=2, marker="o", markerfacecolor="plum", markersize=5)

    def plot_2d_config(path=None, plannerClass=None, ax=None):
        PRMPlotter.plot_2d_obstacle(plannerClass.simulator, ax)
        PRMPlotter.plot_2d_roadmap(plannerClass.nodes, ax)
        if path:
            PRMPlotter.plot_2d_path(path, ax)
            PRMPlotter.plot_2d_state_configuration(path[0], path[-1], ax)
