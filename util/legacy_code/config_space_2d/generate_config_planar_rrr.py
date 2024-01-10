""" Generate Configuration Space for Planar Robot RR in 2D
- Robot Type : Planar RRR
- DOF : 3
- Taskspace map : Geometry Format
- Collision Check : Geometry Based
- Theta 1 in
- Theta 2 in
- Theta 3 in
"""

import os
import sys

wd = os.path.abspath(os.getcwd())
sys.path.append(str(wd))

import numpy as np
from spatial_geometry.spatial_shape import ShapeLine2D, ShapeCollision


def configuration_generate_plannar_rrr(robot, obsList):
    gridSize = 75
    theta1 = np.linspace(-np.pi, np.pi, gridSize)
    theta2 = np.linspace(-np.pi, np.pi, gridSize)
    theta3 = np.linspace(-np.pi, np.pi, gridSize)

    gridMap = np.zeros((gridSize, gridSize, gridSize))

    if obsList == []:
        return 1 - gridMap

    else:
        for th1 in range(len(theta1)):
            for th2 in range(len(theta2)):
                for th3 in range(len(theta3)):
                    print(f"at theta1 {theta1[th1]} | at theta2 {theta2[th2]} | at theta3 {theta3[th3]}")
                    theta = np.array([[theta1[th1]], [theta2[th2]], [theta3[th3]]])
                    linkPose = robot.forward_kinematic(theta, return_link_pos=True)
                    linearm1 = ShapeLine2D(linkPose[0][0], linkPose[0][1], linkPose[1][0], linkPose[1][1])
                    linearm2 = ShapeLine2D(linkPose[1][0], linkPose[1][1], linkPose[2][0], linkPose[2][1])
                    linearm3 = ShapeLine2D(linkPose[2][0], linkPose[2][1], linkPose[3][0], linkPose[3][1])

                    col = []
                    for i in obsList:
                        col1 = ShapeCollision.intersect_line_v_rectangle(linearm1, i)
                        col2 = ShapeCollision.intersect_line_v_rectangle(linearm2, i)
                        col3 = ShapeCollision.intersect_line_v_rectangle(linearm3, i)
                        col.extend((col1, col2, col3))

                    if True in col:
                        gridMap[th1, th2, th3] = 1

        return 1 - gridMap


def configuration_generate_plannar_rrr_first_2joints(robot, obsList):
    gridSize = 75
    theta1 = np.linspace(-np.pi, np.pi, gridSize)
    theta2 = np.linspace(-np.pi, np.pi, gridSize)
    theta3 = np.pi  # assumed fixed

    gridMap = np.zeros((gridSize, gridSize, gridSize))

    for th1 in range(len(theta1)):
        for th2 in range(len(theta2)):
            print(f"at theta1 {theta1[th1]} | at theta2 {theta2[th2]}")
            theta = np.array([[theta1[th1]], [theta2[th2]], [theta3]])
            linkPose = robot.forward_kinematic(theta, return_link_pos=True)
            linearm1 = ShapeLine2D(linkPose[0][0], linkPose[0][1], linkPose[1][0], linkPose[1][1])
            linearm2 = ShapeLine2D(linkPose[1][0], linkPose[1][1], linkPose[2][0], linkPose[2][1])

            col = []
            for i in obsList:
                col1 = ShapeCollision.intersect_line_v_rectangle(linearm1, i)
                col2 = ShapeCollision.intersect_line_v_rectangle(linearm2, i)
                col.extend((col1, col2))

            if True in col:
                gridMap[th1, th2] = 1

    return 1 - gridMap


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from spatial_geometry.taskmap_geo_format import task_rectangle_obs_3, task_rectangle_obs_6
    from robot.planar_rrr import PlanarRRR

    # SECTION - robot class
    r = PlanarRRR()
    obsList = task_rectangle_obs_6()

    # SECTION - plot task space
    r.plot_arm(np.array([0, 0, 0]).reshape(3, 1), plt_basis=True)
    for obs in obsList:
        obs.plot()
    plt.show()

    # SECTION - configuration space
    map = configuration_generate_plannar_rrr(r, obsList)
    # ax = plt.figure().add_subplot(projection='3d')
    # ax.voxels(map, edgecolor='k')
    plt.imshow(map[0, :, :])  # plt view of each index slice 3D into 2D image
    plt.show()