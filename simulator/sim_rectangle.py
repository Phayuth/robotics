import os
import sys

wd = os.path.abspath(os.getcwd())
sys.path.append(str(wd))

import numpy as np
from spatial_geometry.spatial_shape import ShapeRectangle, ShapeCollision
from spatial_geometry.map_class import CostMapClass, CostMapLoader


class TaskSpace2DSimulator:

    def __init__(self) -> None:
        # required for planner
        self.configLimit = [[-np.pi, np.pi], [-np.pi, np.pi]]
        self.configDoF = len(self.configLimit)

        loader = CostMapLoader.loadsave(maptype="task", mapindex=0, reverse=False)
        costMap = CostMapClass(loader, maprange=self.configLimit)
        self.taskMapObs = costMap.costmap2geo(free_space_value=1)

    def collision_check(self, xNewConfig):
        robot = ShapeRectangle(xNewConfig[0,0], xNewConfig[1,0], 0.1, 0.1)
        for obs in self.taskMapObs:
            if ShapeCollision.intersect_rectangle_v_rectangle(robot, obs):
                return True
        return False

    def get_cspace_grid(self):
        sample = np.linspace(-np.pi, np.pi, 360)
        grid = []
        gridRow = []
        for i in sample:
            for j in sample:
                recMoving = ShapeRectangle(i, j, h=0.1, w=0.1)
                col = []
                for k in self.taskMapObs:
                    collision = ShapeCollision.intersect_rectangle_v_rectangle(recMoving, k)
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

    def play_back_path(self, path, axis):
        raise NotImplementedError


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    env = TaskSpace2DSimulator()
    env.plot_cspace()
    plt.show()
