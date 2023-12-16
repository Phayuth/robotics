import os
import sys

wd = os.path.abspath(os.getcwd())
sys.path.append(str(wd))

import numpy as np
from geometry.geometry_class import ObjLine2D, ObjRectangle, CollisionGeometry
from robot.planar_rr import PlanarRR
from map.taskmap_geo_format import task_rectangle_obs_1
from map.mapclass import CostMapClass, CostMapLoader


class RobotArm2DEnvironment:

    def __init__(self):
        # required for planner
        self.configLimit = [[-2*np.pi, 2*np.pi], [-2*np.pi, 2*np.pi]]
        self.configDoF = len(self.configLimit)

        self.robot = PlanarRR()
        self.taskMapObs = task_rectangle_obs_1()

    def collision_check(self, xNewConfig):
        linkPose = self.robot.forward_kinematic(xNewConfig, return_link_pos=True)
        linearm1 = ObjLine2D(linkPose[0][0], linkPose[0][1], linkPose[1][0], linkPose[1][1])
        linearm2 = ObjLine2D(linkPose[1][0], linkPose[1][1], linkPose[2][0], linkPose[2][1])

        link = [linearm1, linearm2]

        for obs in self.taskMapObs:
            for i in range(len(link)):
                if CollisionGeometry.intersect_line_v_rectangle(link[i], obs):
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
            linearm1 = ObjLine2D(linkPose[0][0], linkPose[0][1], linkPose[1][0], linkPose[1][1])

            for i in self.taskMapObs:
                if CollisionGeometry.intersect_line_v_rectangle(linearm1, i):
                    gridMap[0:len(theta2), th1] = 1
                    continue

                else:
                    for th2 in range(len(theta2)):
                        print(f"Theta1 {theta1[th1]} | Theta 2 {theta2[th2]}")
                        theta = np.array([[theta1[th1]], [theta2[th2]]])
                        linkPose = self.robot.forward_kinematic(theta, return_link_pos=True)
                        linearm2 = ObjLine2D(linkPose[1][0], linkPose[1][1], linkPose[2][0], linkPose[2][1])

                        if CollisionGeometry.intersect_line_v_rectangle(linearm2, i):
                            gridMap[th2, th1] = 1

        return 1 - gridMap

    def plot_taskspace(self, theta):
        self.robot.plot_arm(theta, plt_basis=True)
        for obs in self.taskMapObs:
            obs.plot()

    def plot_cspace(self):
        jointRange = np.linspace(-2*np.pi, 2*np.pi, 720)
        collisionPoint = []
        for theta1 in jointRange:
            for theta2 in jointRange:
                node = np.array([[theta1], [theta2]])
                result = self.collision_check(node)
                if result is True:
                    collisionPoint.append([theta1, theta2])

        collisionPoint = np.array(collisionPoint)
        plt.plot(collisionPoint[:, 0], collisionPoint[:, 1], color='darkcyan', linewidth=0, marker='o', markerfacecolor='darkcyan', markersize=1.5)


class TaskSpace2DEnvironment:

    def __init__(self) -> None:
        # required for planner
        self.configLimit = [[-np.pi, np.pi], [-np.pi, np.pi]]
        self.configDoF = len(self.configLimit)

        loader = CostMapLoader.loadsave(maptype="task", mapindex=0, reverse=False)
        costMap = CostMapClass(loader, maprange=self.configLimit)
        self.taskMapObs = costMap.costmap2geo(free_space_value=1)

    def collision_check(self, xNewConfig):
        robot = ObjRectangle(xNewConfig[0,0], xNewConfig[1,0], 0.1, 0.1)
        for obs in self.taskMapObs:
            if CollisionGeometry.intersect_rectangle_v_rectangle(robot, obs):
                return True
        return False

    def get_cspace_grid(self):
        sample = np.linspace(-np.pi, np.pi, 360)
        grid = []
        gridRow = []
        for i in sample:
            for j in sample:
                recMoving = ObjRectangle(i, j, h=2, w=2)
                col = []
                for k in self.taskMapObs:
                    collision = CollisionGeometry.intersect_rectangle_v_rectangle(recMoving, k)
                    col.append(collision)
                if True in col:
                    gridRow.append(True)
                else:
                    gridRow.append(False)
            grid.append(gridRow)
            gridRow = []
        gridNp = np.array(grid).astype(int)
        return 1 - gridNp

    def plot_taskspace(self):
        for obs in self.taskMapObs:
            obs.plot()

    def plot_cspace(self):
        jointRange = np.linspace(-np.pi, np.pi, 360)
        collisionPoint = []
        for theta1 in jointRange:
            for theta2 in jointRange:
                node = np.array([[theta1], [theta2]])
                result = self.collision_check(node)
                if result is True:
                    collisionPoint.append([theta1, theta2])

        collisionPoint = np.array(collisionPoint)
        plt.plot(collisionPoint[:, 0], collisionPoint[:, 1], color='darkcyan', linewidth=0, marker='o', markerfacecolor='darkcyan', markersize=1.5)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    env = RobotArm2DEnvironment()
    env.plot_cspace()
    plt.show()


    # gridNp = env.generate_cspace()
    # plt.imshow(gridNp)
    # plt.show()


    # env = TaskSpace2DEnvironment()
    # env.plot_cspace()
    # plt.show()
