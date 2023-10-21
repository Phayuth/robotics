""" Generate Configuration Space for Planar Robot RR in 2D
- Robot Type : Planar RR
- DOF : 2
- Taskspace map : Geometry Format
- Collision Check : Geometry Based
- Theta 1 in Column
- Theta 2 in Row
"""

import os
import sys
import time
wd = os.path.abspath(os.getcwd())
sys.path.append(str(wd))

import numpy as np
from geometry.geometry_class import ObjLine2D, CollisionGeometry


def configuration_generate_plannar_rr(robot, obsList):
    gridSize = 360
    theta1 = np.linspace(-np.pi, np.pi, gridSize)
    theta2 = np.linspace(-np.pi, np.pi, gridSize)

    gridMap = np.zeros((gridSize, gridSize))

    for th1 in range(len(theta1)):
        theta = np.array([[theta1[th1]], [0]])
        linkPose = robot.forward_kinematic(theta, return_link_pos=True)
        linearm1 = ObjLine2D(linkPose[0][0], linkPose[0][1], linkPose[1][0], linkPose[1][1])

        for i in obsList:
            if CollisionGeometry.intersect_line_v_rectangle(linearm1, i):
                gridMap[0:len(theta2), th1] = 1
                continue

            else:
                for th2 in range(len(theta2)):
                    print(f"Theta1 {theta1[th1]} | Theta 2 {theta2[th2]}")
                    theta = np.array([[theta1[th1]], [theta2[th2]]])
                    linkPose = robot.forward_kinematic(theta, return_link_pos=True)
                    linearm2 = ObjLine2D(linkPose[1][0], linkPose[1][1], linkPose[2][0], linkPose[2][1])

                    if CollisionGeometry.intersect_line_v_rectangle(linearm2, i):
                        gridMap[th2, th1] = 1

    return 1 - gridMap


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from map.mapclass import map_vec
    from map.taskmap_geo_format import task_rectangle_obs_1
    from robot.planar_rr import PlanarRR

    # SECTION - robot class
    r = PlanarRR()
    desiredPose = np.array([[3.5], [1.8]])
    theta = r.inverse_kinematic_geometry(desiredPose, elbow_option=0)
    thetaIndex = map_vec(theta, -np.pi, np.pi, 0, 360).astype(int)  # map theta to index image (option elbow up)
    thetaVal = map_vec(thetaIndex, 0, 360, -np.pi, np.pi)  # map index image to theta (option elbow up)
    obsList = task_rectangle_obs_1()

    # SECTION - plot task space
    r.plot_arm(theta, plt_basis=True)
    for obs in obsList:
        obs.plot()
    plt.show()

    # SECTION - configuration space
    time_start = time.perf_counter()
    gridNp = configuration_generate_plannar_rr(r, obsList)
    print(f"time elapsed {time.perf_counter() - time_start}") # first method 36sec, second 39sec
    gridNp[thetaIndex[1, 0], thetaIndex[0, 0]] = 2  # add a point in index of grid

    # SECTION - plot configspace in image format
    plt.imshow(gridNp)
    plt.grid(True)
    plt.show()