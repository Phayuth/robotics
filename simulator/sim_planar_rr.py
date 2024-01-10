import os
import sys

wd = os.path.abspath(os.getcwd())
sys.path.append(str(wd))

import numpy as np
from spatial_geometry.spatial_shape import ShapeLine2D, ShapeCollision
from robot.nonmobile.planar_rr import PlanarRR
from spatial_geometry.taskmap_geo_format import NonMobileTaskMap


class RobotArm2DSimulator:

    def __init__(self):
        # required for planner
        self.configLimit = [[-2*np.pi, 2*np.pi], [-2*np.pi, 2*np.pi]]
        self.configDoF = len(self.configLimit)

        self.robot = PlanarRR()
        self.taskMapObs = NonMobileTaskMap.task_rectangle_obs_1()

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
        gridSize = 720
        theta1 = np.linspace(-2*np.pi, 2*np.pi, gridSize)
        theta2 = np.linspace(-2*np.pi, 2*np.pi, gridSize)

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

    def plot_taskspace(self, theta):
        self.robot.plot_arm(theta, plt_basis=True)
        for obs in self.taskMapObs:
            obs.plot()

    def plot_cspace(self, axis):
        jointRange = np.linspace(-2*np.pi, 2*np.pi, 720)
        collisionPoint = []
        for theta1 in jointRange:
            for theta2 in jointRange:
                node = np.array([[theta1], [theta2]])
                result = self.collision_check(node)
                if result is True:
                    collisionPoint.append([theta1, theta2])

        collisionPoint = np.array(collisionPoint)
        axis.plot(collisionPoint[:, 0], collisionPoint[:, 1], color='darkcyan', linewidth=0, marker='o', markerfacecolor='darkcyan', markersize=1.5)

    def play_back_path(self, path, axis):
        raise NotImplementedError


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    env = RobotArm2DSimulator()
    env.plot_cspace()
    plt.show()

    # gridNp = env.generate_cspace()
    # plt.imshow(gridNp)
    # plt.show()