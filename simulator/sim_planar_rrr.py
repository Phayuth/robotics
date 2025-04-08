import os
import sys

sys.path.append(str(os.path.abspath(os.getcwd())))

import numpy as np
import matplotlib.pyplot as plt
from spatial_geometry.spatial_shape import ShapeLine2D, ShapeCollision
from robot.nonmobile.planar_rrr import PlanarRRR


class RobotArm2DRRRSimulator:

    def __init__(self, taskspace, torusspace=False):
        # required for planner
        self.toruspace = torusspace
        if self.toruspace:
            self.configLimit = [[-2 * np.pi, 2 * np.pi], [-2 * np.pi, 2 * np.pi], [-2 * np.pi, 2 * np.pi]]
        else:
            self.configLimit = [[-np.pi, np.pi], [-np.pi, np.pi], [-np.pi, np.pi]]

        self.configDoF = len(self.configLimit)

        self.robot = PlanarRRR()
        self.taskspace = taskspace
        self.taskMapObs = self.taskspace.task_map()

        # ICROS 2023 confpaper
        # xTarg = 1.5
        # yTarg = 0.2
        # alphaTarg = 2  # given from grapse pose candidate
        # hD = 0.25
        # wD = 0.25
        # rCrop = 0.1
        # phiTarg = alphaTarg - np.pi
        # xTopStart = (rCrop + hD) * np.cos(alphaTarg - np.pi / 2) + xTarg
        # yTopStart = (rCrop + hD) * np.sin(alphaTarg - np.pi / 2) + yTarg
        # xBotStart = (rCrop) * np.cos(alphaTarg + np.pi / 2) + xTarg
        # yBotStart = (rCrop) * np.sin(alphaTarg + np.pi / 2) + yTarg
        # recTop = ShapeRectangle(xTopStart, yTopStart, hD, wD, angle=alphaTarg)
        # recBot = ShapeRectangle(xBotStart, yBotStart, hD, wD, angle=alphaTarg)
        # self.taskMapObs = [recTop, recBot]
        # target = np.array([xTarg, yTarg, phiTarg]).reshape(3, 1)
        # thetaGoal = self.inverse_kinematic_geometry(target, elbow_option=0)

    def collision_check(self, xNewConfig):
        linkPose = self.robot.forward_kinematic(xNewConfig, return_link_pos=True)
        linearm1 = ShapeLine2D(linkPose[0][0], linkPose[0][1], linkPose[1][0], linkPose[1][1])
        linearm2 = ShapeLine2D(linkPose[1][0], linkPose[1][1], linkPose[2][0], linkPose[2][1])
        linearm3 = ShapeLine2D(linkPose[2][0], linkPose[2][1], linkPose[3][0], linkPose[3][1])

        link = [linearm1, linearm2, linearm3]

        for obs in self.taskMapObs:
            for i in range(len(link)):
                if ShapeCollision.intersect_line_v_rectangle(link[i], obs):
                    return True
        return False

    def get_cspace_grid(self):
        raise NotImplementedError("This robot is 3DOF. It is difficult to view the 3D grid.")

    def plot_taskspace(self):
        for obs in self.taskMapObs:
            obs.plot("#2ca08980")

    def plot_cspace(self, axis):
        raise NotImplementedError("This robot is 3DOF. It is difficult to view the 3D grid.")

    def play_back_path(self, path, animation):
        """
        path shape must be in (3,n)
        """
        # plot task space
        fig, ax = plt.subplots()
        # ax.grid(True)
        ax.set_aspect("equal")
        ax.set_xlim(self.taskspace.xlim)
        ax.set_ylim(self.taskspace.ylim)
        self.plot_taskspace()

        # plot animation link
        (robotLinks,) = ax.plot([], [], color=self.robot.linkcolor, linewidth=self.robot.linkwidth, marker=self.robot.jointmarker, markerfacecolor="r")

        def update(frame):
            link = self.robot.forward_kinematic(path[:, frame].reshape(3, 1), return_link_pos=True)
            robotLinks.set_data([link[0][0], link[1][0], link[2][0], link[3][0]], [link[0][1], link[1][1], link[2][1], link[3][1]])

        animation = animation.FuncAnimation(fig, update, frames=path.shape[1], interval=100, repeat=False)
        plt.show()

    def plot_view(self, thetas, shadowseq=False, colors=[]):
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
        self.robot.plot_arm(thetas, ax, shadow=shadowseq, colors=colors)
        plt.show()
