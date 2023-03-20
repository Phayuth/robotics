import os
import sys
wd = os.path.abspath(os.getcwd())
sys.path.append(str(wd))

import matplotlib.pyplot as plt
from collision_check_geometry import collision_class
from map.taskmap_geo_format import task_rectangle_obs_1
from robot.planar_rr import planar_rr
import numpy as np

def configuration_generate_plannar_rr(robot, obs_list):

    grid_size = 360
    theta1 = np.linspace(-np.pi, np.pi, grid_size)
    theta2 = np.linspace(-np.pi, np.pi, grid_size)

    # config_array = []
    # config_array_row = []
    grid_map = np.zeros((grid_size, grid_size))

    for j in range(len(theta1)):
        for k in range(len(theta2)):
            print(f"at theta1 {theta1[j]} | at theta2 {theta2[k]}")
            theta = np.array([[theta1[j]],[theta2[k]]])
            link_pose = robot.forward_kinematic(theta, return_link_pos=True)
            linearm1 = collision_class.line_obj(link_pose[0][0], link_pose[0][1], link_pose[1][0], link_pose[1][1])
            linearm2 = collision_class.line_obj(link_pose[1][0], link_pose[1][1], link_pose[2][0], link_pose[2][1])

            col = []
            for i in obs_list:
                col1 = collision_class.line_v_rectangle(linearm1, i)
                col2 = collision_class.line_v_rectangle(linearm2, i)
                col.extend((col1,col2))

            if True in col:
                grid_map[j,k] = 1

    return 1 - grid_map

if __name__=="__main__":
    r = planar_rr()
    obs = task_rectangle_obs_1()
    grid_np = configuration_generate_plannar_rr(r, obs)
    plt.imshow(grid_np)
    plt.show()