import os
import sys

sys.path.append(str(os.path.abspath(os.getcwd())))

import numpy as np
import matplotlib.pyplot as plt
from spatial_geometry.spatial_shape import ShapeCircle, ShapeCollision
from spatial_geometry.taskmap_geo_format import MobileTaskMap
from robot.mobile.differential import DifferentialDrive


class DiffDrive2DSimulator:

    def __init__(self, env=1) -> None:
        # required for planner
        if env == 1:
            self.configLimit = [[0.0, 30.0], [0.0, 17.5]]
            self.taskMapObs = MobileTaskMap.warehouse_map()
        elif env == 2:
            self.configLimit = [[-10.0, 10.0], [-10.0, 10.0]]
            self.taskMapObs = MobileTaskMap.no_collision_map()

        self.configDoF = len(self.configLimit)
        self.robot = DifferentialDrive(wheelRadius=0.33, baseLength=0.7, baseWidth=1) # scout 2.0

    def collision_check(self, xNewConfig):
        baseCollison = ShapeCircle(xNewConfig[0, 0], xNewConfig[1, 0], self.robot.collisionR)
        for obs in self.taskMapObs:
            if ShapeCollision.intersect_circle_v_rectangle(baseCollison, obs):
                return True
        return False

    def get_cspace_grid(self):
        raise NotImplementedError

    def plot_taskspace(self):
        for obs in self.taskMapObs:
            obs.plot()

    def plot_cspace(self):
        xRange = np.linspace(self.configLimit[0][0], self.configLimit[0][1], 500)
        yRange = np.linspace(self.configLimit[1][0], self.configLimit[1][1], 500)
        collisionPoint = []
        for x in xRange:
            for y in yRange:
                node = np.array([[x], [y]])
                if self.collision_check(node) is True:
                    collisionPoint.append([x, y])

        collisionPoint = np.array(collisionPoint)
        plt.plot(collisionPoint[:, 0], collisionPoint[:, 1], color="darkcyan", linewidth=0, marker="o", markerfacecolor="darkcyan", markersize=1.5)

    def play_back_path(self, path, animation):  # path format (3,n)
        # plot task space
        fig, ax = plt.subplots()
        ax.grid(True)
        ax.set_aspect("equal")
        ax.set_xlim((self.configLimit[0][0], self.configLimit[0][1]))
        ax.set_ylim((self.configLimit[1][0], self.configLimit[1][1]))
        self.plot_taskspace()

        # plot path
        ax.plot(path[0, :], path[1, :], "--", color="grey")

        # plot animation link
        (link1,) = ax.plot([], [], "teal")
        (link2,) = ax.plot([], [], "olive")
        (link3,) = ax.plot([], [], "teal")
        (link4,) = ax.plot([], [], "olive")
        (link5,) = ax.plot([], [], "olive")
        (link6,) = ax.plot([], [], "teal")

        def update(frame):
            link = self.robot.robot_link(path[:, frame].reshape(3, 1))
            link1.set_data([link[0][0], link[2][0]], [link[0][1], link[2][1]])
            link2.set_data([link[1][0], link[2][0]], [link[1][1], link[2][1]])
            link3.set_data([link[2][0], link[3][0]], [link[2][1], link[3][1]])
            link4.set_data([link[3][0], link[4][0]], [link[3][1], link[4][1]])
            link5.set_data([link[4][0], link[1][0]], [link[4][1], link[1][1]])
            link6.set_data([link[0][0], link[3][0]], [link[0][1], link[3][1]])

        animation = animation.FuncAnimation(fig, update, frames=(path.shape[1]), interval=1)
        plt.show()

    def path_generator(self):
        plt.xlim((self.configLimit[0][0], self.configLimit[0][1]))
        plt.ylim((self.configLimit[1][0], self.configLimit[1][1]))
        plt.grid(True)
        points = []

        def onclick(event):
            if event.button == 1:
                points.append([event.xdata, event.ydata])
                if len(points) == 1:
                    plt.plot(points[0][0], points[0][0])
                elif len(points) > 1:
                    plt.plot([points[-1][0], points[-2][0]], [points[-1][1], points[-2][1]])
                plt.draw()

        plt.gcf().canvas.mpl_connect("button_press_event", onclick)
        plt.show()
        return points


if __name__ == "__main__":
    from matplotlib import animation
    from datasave.joint_value.pre_record_value import PreRecordedPathMobileRobot

    env = DiffDrive2DSimulator()
    env.plot_taskspace()
    point = env.path_generator()
    print(f"> point: \n{point}")

    xy = np.array(PreRecordedPathMobileRobot.warehouse_path)
    th = np.full((203, 1), 0)
    path = np.hstack((xy, th)).T
    env.play_back_path(path, animation)
