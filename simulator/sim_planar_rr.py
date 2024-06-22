import os
import sys

wd = os.path.abspath(os.getcwd())
sys.path.append(str(wd))

import numpy as np
import matplotlib.pyplot as plt
from spatial_geometry.spatial_shape import ShapeLine2D, ShapeCollision
from robot.nonmobile.planar_rr import PlanarRR
from spatial_geometry.taskmap_geo_format import NonMobileTaskMap


class RobotArm2DSimulator:

    def __init__(self, torusspace=False):
        # required for planner
        self.torusspace = torusspace
        if self.torusspace:
            self.configLimit = [[-2*np.pi, 2*np.pi], [-2*np.pi, 2*np.pi]]
        else:
            self.configLimit = [[-np.pi, np.pi], [-np.pi, np.pi]]

        self.configDoF = len(self.configLimit)

        self.robot = PlanarRR()
        # self.taskMapObs = NonMobileTaskMap.task_rectangle_obs_1()
        self.taskMapObs = NonMobileTaskMap.paper_torus_exp()
        # self.taskMapObs = NonMobileTaskMap.thesis_exp()

    def collision_check(self, xNewConfig):
        linkPose = self.robot.forward_kinematic(xNewConfig, return_link_pos=True)
        linearm1 = ShapeLine2D(linkPose[0][0], linkPose[0][1], linkPose[1][0], linkPose[1][1])
        linearm2 = ShapeLine2D(linkPose[1][0], linkPose[1][1], linkPose[2][0], linkPose[2][1])

        link = [linearm1, linearm2]

        for obs in self.taskMapObs:
            for i in range(len(link)):
                if ShapeCollision.intersect_line_v_rectangle(link[i], obs):
                    return True
        return False

    def get_cspace_grid(self): #generate into 2d array plot by imshow
        if self.torusspace:
            gridSize = 720
            theta1 = np.linspace(-2*np.pi, 2*np.pi, gridSize)
            theta2 = np.linspace(-2*np.pi, 2*np.pi, gridSize)
        else:
            gridSize = 360
            theta1 = np.linspace(-np.pi, np.pi, gridSize)
            theta2 = np.linspace(-np.pi, np.pi, gridSize)

        gridMap = np.zeros((gridSize, gridSize))

        for th1 in range(len(theta1)):
            theta = np.array([[theta1[th1]], [0]])
            linkPose = self.robot.forward_kinematic(theta, return_link_pos=True)
            linearm1 = ShapeLine2D(linkPose[0][0], linkPose[0][1], linkPose[1][0], linkPose[1][1])

            for i in self.taskMapObs:
                if ShapeCollision.intersect_line_v_rectangle(linearm1, i):
                    gridMap[0:len(theta2), th1] = 1
                    continue

                else:
                    for th2 in range(len(theta2)):
                        print(f"Theta1 {theta1[th1]} | Theta 2 {theta2[th2]}")
                        theta = np.array([[theta1[th1]], [theta2[th2]]])
                        linkPose = self.robot.forward_kinematic(theta, return_link_pos=True)
                        linearm2 = ShapeLine2D(linkPose[1][0], linkPose[1][1], linkPose[2][0], linkPose[2][1])

                        if ShapeCollision.intersect_line_v_rectangle(linearm2, i):
                            gridMap[th2, th1] = 1

        return 1 - gridMap

    def plot_taskspace(self):
        for obs in self.taskMapObs:
            obs.plot()

    def plot_cspace(self, axis):
        if self.torusspace:
            jointRange = np.linspace(-2*np.pi, 2*np.pi, 720)
        else:
            jointRange = np.linspace(-np.pi, np.pi, 360)

        collisionPoint = []
        for theta1 in jointRange:
            for theta2 in jointRange:
                node = np.array([[theta1], [theta2]])
                result = self.collision_check(node)
                if result is True:
                    collisionPoint.append([theta1, theta2])

        collisionPoint = np.array(collisionPoint)
        axis.plot(collisionPoint[:, 0], collisionPoint[:, 1], color='darkcyan', linewidth=0, marker='o', markerfacecolor='darkcyan', markersize=1.5)

    def play_back_path(self, path, animation):  # path format (2,n)
        # plot task space
        fig, ax = plt.subplots()
        ax.grid(True)
        ax.set_aspect("equal")
        ax.set_xlim(-5, 5)
        ax.set_ylim(-5, 5)
        self.plot_taskspace()

        # plot animation link
        robotLinks, = ax.plot([], [], color='indigo', linewidth=5, marker='o', markerfacecolor='r')

        def update(frame):
            link = self.robot.forward_kinematic(path[:, frame].reshape(2,1), return_link_pos=True)
            robotLinks.set_data([link[0][0], link[1][0], link[2][0]], [link[0][1], link[1][1], link[2][1]])

        animation = animation.FuncAnimation(fig, update, frames=(path.shape[1]), interval=1)
        plt.show()

    def plot_view(self, thetas):
        # plot task space
        fig, ax = plt.subplots()
        ax.grid(True)
        ax.set_aspect("equal")
        # ax.set_xlim(-3, 4)
        # ax.set_ylim(-0.5, 5)
        ax.set_xlim(-4, 4)
        ax.set_ylim(-4, 4)
        self.plot_taskspace()

        for the in thetas:
            self.robot.plot_arm(the.reshape(2,1), ax)
        return ax


if __name__ == "__main__":
    from matplotlib import animation
    import matplotlib.pyplot as plt

    env = RobotArm2DSimulator(torusspace=True)

    # fig, ax = plt.subplots(1,1)
    # env.plot_cspace(ax)
    # plt.show()

    # t1 = np.deg2rad(np.arange(0,90,1))
    # t2 = np.deg2rad(np.arange(0,90,1))
    # tt = np.vstack((t1,t2))
    # env.play_back_path(tt, animation)

    thetas = np.array([[0,0], [2.1, 1.5], [2, 1.5], [2.2, 1.5]])
    print(f"> thetas.shape: {thetas.shape}")
    print(f"> thetas: {thetas}")
    env.plot_view(thetas)
    plt.show()