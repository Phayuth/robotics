import os
import sys

wd = os.path.abspath(os.getcwd())
sys.path.append(str(wd))

import numpy as np
from spatial_geometry.spatial_shape import ShapeCircle, ShapeCollision
from spatial_geometry.taskmap_geo_format import MobileTaskMap
from robot.mobile.differential import DifferentialDrive


class DiffDrive2DSimulator:

    def __init__(self) -> None:
        # required for planner
        self.configLimit = [[0.0, 30.0], [0.0, 17.5]]
        self.configDoF = len(self.configLimit)

        self.robot = DifferentialDrive(wheelRadius=0.1, baseLength=0.7, baseWidth=1)
        self.taskMapObs = MobileTaskMap.warehouse_map()

    def collision_check(self, xNewConfig):
        baseCollison = ShapeCircle(xNewConfig[0,0], xNewConfig[1,0], self.robot.collisionR)
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
        plt.plot(collisionPoint[:, 0], collisionPoint[:, 1], color='darkcyan', linewidth=0, marker='o', markerfacecolor='darkcyan', markersize=1.5)

    def play_back_path(self, path, axis):
        # self.set_start_joint_value(path[0])
        # self.set_goal_joint_value(path[-1])
        # self.set_aux_joint_value(path[-2])
        for p in path:
            self.robot.plot_robot(p.config, axis)

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
                    plt.plot([points[-1][0] , points[-2][0]], [points[-1][1] , points[-2][1]])
                plt.draw()
        plt.gcf().canvas.mpl_connect('button_press_event', onclick)
        plt.show()
        return points

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    env = DiffDrive2DSimulator()
    env.plot_taskspace()
    point = env.path_generator()
    print(f"==>> point: \n{point}")
    plt.show()
