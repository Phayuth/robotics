import os
import sys

wd = os.path.abspath(os.getcwd())
sys.path.append(str(wd))

import numpy as np
from collision_check_geometry.collision_class import ObjLine2D, intersect_line_v_rectangle
from robot.planar_rr import PlanarRR
from map.taskmap_geo_format import task_rectangle_obs_1


class RobotArm2DEnvironment:

    def __init__(self) -> None:
        self.robot = PlanarRR()
        self.taskMapObs = task_rectangle_obs_1()

    def check_collision(self, xNew):
        theta = np.array([xNew.x, xNew.y]).reshape(2, 1)
        linkPose = self.robot.forward_kinematic(theta, return_link_pos=True)
        linearm1 = ObjLine2D(linkPose[0][0], linkPose[0][1], linkPose[1][0], linkPose[1][1])
        linearm2 = ObjLine2D(linkPose[1][0], linkPose[1][1], linkPose[2][0], linkPose[2][1])

        link = [linearm1, linearm2]

        for obs in self.taskMapObs:
            for i in range(len(link)):
                if intersect_line_v_rectangle(link[i], obs):
                    return True
        return False


if __name__ == "__main__":
    from planar_rrt_component import Node
    import matplotlib.pyplot as plt

    env = RobotArm2DEnvironment()

    jointRange = np.linspace(-np.pi, np.pi, 360)
    xy_points = []
    for theta1 in jointRange:
        for theta2 in jointRange:
            result = env.check_collision(Node(theta1, theta2))
            if result is True:
                xy_points.append([theta1, theta2])

    xy_points = np.array(xy_points)
    plt.plot(xy_points[:, 0], xy_points[:, 1], color='tomato', linewidth=0, marker = 'o', markerfacecolor = 'tomato', markersize=1.5)
    plt.show()