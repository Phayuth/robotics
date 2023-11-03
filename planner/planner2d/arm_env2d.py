import os
import sys

wd = os.path.abspath(os.getcwd())
sys.path.append(str(wd))

import numpy as np
from geometry.geometry_class import ObjLine2D, ObjRectangle, CollisionGeometry
from robot.planar_rr import PlanarRR
from map.taskmap_geo_format import task_rectangle_obs_1

from map.mapclass import CostMapClass, CostMapLoader
from map.taskmap_img_format import map_2d_1

class RobotArm2DEnvironment:

    def __init__(self) -> None:
        self.robot = PlanarRR()
        self.taskMapObs = task_rectangle_obs_1()

    def collision_check(self, xNew):
        theta = xNew.config
        linkPose = self.robot.forward_kinematic(theta, return_link_pos=True)
        linearm1 = ObjLine2D(linkPose[0][0], linkPose[0][1], linkPose[1][0], linkPose[1][1])
        linearm2 = ObjLine2D(linkPose[1][0], linkPose[1][1], linkPose[2][0], linkPose[2][1])

        link = [linearm1, linearm2]

        for obs in self.taskMapObs:
            for i in range(len(link)):
                if CollisionGeometry.intersect_line_v_rectangle(link[i], obs):
                    return True
        return False


class TaskSpace2DEnvironment:

    def __init__(self) -> None:
        loader = CostMapLoader.loadsave(maptype="task", mapindex=0, reverse=False)
        costMap = CostMapClass(loader, maprange=[[-np.pi, np.pi], [-np.pi, np.pi]])
        self.taskMapObs = costMap.costmap2geo(free_space_value=1)

    def collision_check(self, xNew):
        theta = xNew.config
        robot = ObjRectangle(theta[0,0], theta[1,0], 0.1, 0.1)
        for obs in self.taskMapObs:
            if CollisionGeometry.intersect_rectangle_v_rectangle(robot, obs):
                return True
        return False


if __name__ == "__main__":
    from planner.rrt_component import Node
    import matplotlib.pyplot as plt

    # env = RobotArm2DEnvironment()
    env = TaskSpace2DEnvironment()

    jointRange = np.linspace(-np.pi, np.pi, 360)
    xy_points = []
    for theta1 in jointRange:
        for theta2 in jointRange:
            node = Node(np.array([theta1, theta2]).reshape(2,1))
            result = env.collision_check(node)
            if result is True:
                xy_points.append([theta1, theta2])

    xy_points = np.array(xy_points)
    plt.plot(xy_points[:, 0], xy_points[:, 1], color='tomato', linewidth=0, marker = 'o', markerfacecolor = 'tomato', markersize=1.5)
    plt.show()