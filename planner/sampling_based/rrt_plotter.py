import os
import sys

wd = os.path.abspath(os.getcwd())
sys.path.append(str(wd))

import numpy as np
import matplotlib.patches as mpatches


class RRTPlotter:

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
        axis.plot(collisionPoint[:, 0], collisionPoint[:, 1], color='darkcyan', linewidth=0, marker='o', markerfacecolor='darkcyan', markersize=1.5)

    def plot_2d_single_tree(tree, axis):
        for vertex in tree:
            if vertex.parent == None:
                pass
            else:
                axis.plot([vertex.config[0], vertex.parent.config[0]], [vertex.config[1], vertex.parent.config[1]], color="darkgray")

    def plot_2d_node_in_tree(tree, axis):
        for vertex in tree:
            if vertex.parent == None:
                pass
            else:
                axis.plot([vertex.config[0]], [vertex.config[1]], color="blue", linewidth=0, marker='o', markerfacecolor='yellow')

    def plot_2d_dual_tree(tree1, tree2, axis):
        for vertex in tree1:
            if vertex.parent == None:
                pass
            else:
                axis.plot([vertex.config[0], vertex.parent.config[0]], [vertex.config[1], vertex.parent.config[1]], color="darkgray")

        for vertex in tree2:
            if vertex.parent == None:
                pass
            else:
                axis.plot([vertex.config[0], vertex.parent.config[0]], [vertex.config[1], vertex.parent.config[1]], color="darkgray")

    def plot_2d_state_configuration(xStart, xApp, xGoal, axis):
        axis.plot(xStart.config[0], xStart.config[1], color="blue", linewidth=0, marker='o', markerfacecolor='yellow')

        if isinstance(xApp, list):
            for xA in xApp:
                axis.plot(xA.config[0], xA.config[1], color="blue", linewidth=0, marker='o', markerfacecolor='green')
            for xG in xGoal:
                axis.plot(xG.config[0], xG.config[1], color="blue", linewidth=0, marker='o', markerfacecolor='red')
        else:
            axis.plot(xApp.config[0], xApp.config[1], color="blue", linewidth=0, marker='o', markerfacecolor='green')
            axis.plot(xGoal.config[0], xGoal.config[1], color="blue", linewidth=0, marker='o', markerfacecolor='red')

    def plot_2d_path(path, axis):
        axis.plot(path[0,:], path[1,:], color='blue', linewidth=2, marker='o', markerfacecolor='plum', markersize=5)

    def plot_performance(perfMatrix, axis):
        costGraph = perfMatrix["Cost Graph"]
        iteration, costs = zip(*costGraph)

        legendItems = [
            mpatches.Patch(color='blue', label=f'Parameters: eta = [{perfMatrix["Parameters"]["eta"]}]'),
            mpatches.Patch(color='blue', label=f'Parameters: subEta = [{perfMatrix["Parameters"]["subEta"]}]'),
            mpatches.Patch(color='blue', label=f'Parameters: Max Iteration = [{perfMatrix["Parameters"]["Max Iteration"]}]'),
            mpatches.Patch(color='blue', label=f'Parameters: Rewire Radius = [{perfMatrix["Parameters"]["Rewire Radius"]}]'),
            mpatches.Patch(color='red', label=f'# Node = [{perfMatrix["Number of Node"]}]'),
            mpatches.Patch(color='green', label=f'Initial Path Cost = [{perfMatrix["Cost Graph"][0][1]:.5f}]'),
            mpatches.Patch(color='yellow', label=f'Initial Path Found on Iteration = [{perfMatrix["Cost Graph"][0][0]}]'),
            mpatches.Patch(color='pink', label=f'Final Path Cost = [{perfMatrix["Cost Graph"][-1][1]:.5f}]'),
            mpatches.Patch(color='indigo', label=f'Total Planning Time = [{perfMatrix["Total Planning Time"]:.5f}]'),
            mpatches.Patch(color='tan', label=f'Planning Time Only = [{perfMatrix["Planning Time Only"]:.5f}]'),
            mpatches.Patch(color='olive', label=f'KCD Time Spend = [{perfMatrix["KCD Time Spend"]:.5f}]'),
            mpatches.Patch(color='cyan', label=f'# KCD = [{perfMatrix["Number of Collision Check"]}]'),
            mpatches.Patch(color='peru', label=f'Avg KCD Time = [{perfMatrix["Average KCD Time"]:.5f}]'),
        ]

        axis.plot(iteration, costs, color='blue', marker='o', markersize=5)
        axis.legend(handles=legendItems)

        axis.set_xlabel('Iteration')
        axis.set_ylabel('Cost')
        axis.set_title(f'Performance Plot of [{perfMatrix["Planner Name"]}]')

    def plot_2d_config_single_tree(plannerClass, path, ax):
        RRTPlotter.plot_2d_obstacle(plannerClass.simulator, ax)
        RRTPlotter.plot_2d_single_tree(plannerClass.treeVertex, ax)
        if path is not None:
            RRTPlotter.plot_2d_path(path, ax)
        try:
            RRTPlotter.plot_2d_state_configuration(plannerClass.xStart, plannerClass.xApp, plannerClass.xGoal, ax)
        except:
            RRTPlotter.plot_2d_state_configuration(plannerClass.xStart, plannerClass.xAppList, plannerClass.xGoalList, ax)

    def plot_2d_config_dual_tree(plannerClass, path, ax):
        RRTPlotter.plot_2d_obstacle(plannerClass.simulator, ax)
        RRTPlotter.plot_2d_dual_tree(plannerClass.treeVertexStart, plannerClass.treeVertexGoal, ax)
        if path is not None:
            RRTPlotter.plot_2d_path(path, ax)
        try:
            RRTPlotter.plot_2d_state_configuration(plannerClass.xStart, plannerClass.xApp, plannerClass.xGoal, ax)
        except:
            RRTPlotter.plot_2d_state_configuration(plannerClass.xStart, plannerClass.xAppList, plannerClass.xGoalList, ax)
