import os
import sys

sys.path.append(str(os.path.abspath(os.getcwd())))

import numpy as np
import matplotlib.pyplot as plt
from spatial_geometry.spatial_shape import ShapeLine2D, ShapeCollision
from robot.nonmobile.planar_6r import Planar6R


class RobotArm6RSimulator:

    def __init__(self, taskspace, torusspace=False):
        # required for planner
        self.torusspace = torusspace
        if self.torusspace:
            self.configLimit = [
                [-2 * np.pi, 2 * np.pi],
                [-2 * np.pi, 2 * np.pi],
                [-2 * np.pi, 2 * np.pi],
                [-2 * np.pi, 2 * np.pi],
                [-2 * np.pi, 2 * np.pi],
                [-2 * np.pi, 2 * np.pi],
            ]
        else:
            self.configLimit = [
                [-np.pi, np.pi],
                [-np.pi, np.pi],
                [-np.pi, np.pi],
                [-np.pi, np.pi],
                [-np.pi, np.pi],
                [-np.pi, np.pi],
            ]

        self.configDoF = len(self.configLimit)

        self.robot = Planar6R()
        self.taskspace = taskspace
        self.taskMapObs = self.taskspace.task_map()

    def collision_check(self, xNewConfig):
        linkPose = self.robot.forward_kinematic(xNewConfig, return_link_pos=True)
        linearm1 = ShapeLine2D(
            linkPose[0][0], linkPose[0][1], linkPose[1][0], linkPose[1][1]
        )
        linearm2 = ShapeLine2D(
            linkPose[1][0], linkPose[1][1], linkPose[2][0], linkPose[2][1]
        )
        linearm3 = ShapeLine2D(
            linkPose[2][0], linkPose[2][1], linkPose[3][0], linkPose[3][1]
        )
        linearm4 = ShapeLine2D(
            linkPose[3][0], linkPose[3][1], linkPose[4][0], linkPose[4][1]
        )
        linearm5 = ShapeLine2D(
            linkPose[4][0], linkPose[4][1], linkPose[5][0], linkPose[5][1]
        )
        linearm6 = ShapeLine2D(
            linkPose[5][0], linkPose[5][1], linkPose[6][0], linkPose[6][1]
        )

        link = [linearm1, linearm2, linearm3, linearm4, linearm5, linearm6]

        for obs in self.taskMapObs:
            for i in range(len(link)):
                if ShapeCollision.intersect_line_v_rectangle(link[i], obs):
                    return True
        return False

    def plot_taskspace(self):
        for obs in self.taskMapObs:
            obs.plot("#2ca08980")

    def play_back_path(self, path, animation):
        """
        path shape format (2,n)
        """
        # plot task space
        fig, ax = plt.subplots()
        ax.grid(True)
        ax.set_aspect("equal")
        ax.set_xlim(self.taskspace.xlim)
        ax.set_ylim(self.taskspace.ylim)
        self.plot_taskspace()

        # plot animation link
        (robotLinks,) = ax.plot(
            [], [], color="indigo", linewidth=5, marker="o", markerfacecolor="r"
        )

        def update(frame):
            link = self.robot.forward_kinematic(
                path[:, frame].reshape(2, 1), return_link_pos=True
            )
            robotLinks.set_data(
                [link[0][0], link[1][0], link[2][0]],
                [link[0][1], link[1][1], link[2][1]],
            )

        animation = animation.FuncAnimation(
            fig, update, frames=(path.shape[1]), interval=1
        )
        plt.show()

    def plot_view(self, thetas):
        fig, ax = plt.subplots()
        s = 1
        fig.set_size_inches(w=s * 3.40067, h=s * 3.40067)
        fig.tight_layout()
        ax.set_aspect("equal")
        ax.set_xlim(self.taskspace.xlim)
        ax.set_ylim(self.taskspace.ylim)
        ax.axhline(color="gray", alpha=0.4)
        ax.axvline(color="gray", alpha=0.4)
        self.plot_taskspace()

        for the in thetas:
            self.robot.plot_arm(the.reshape(6, 1), ax)
        plt.show()