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

    config_array = []
    config_array_row = []

    for j in theta1:
        for k in theta2:
            print(f"at theta1 {j} | at theta2 {k}")
            theta = np.array([[j],[k]])
            link_pose = robot.forward_kinematic(theta, return_link_pos=True)
            linearm1 = collision_class.line_obj(link_pose[0][0], link_pose[0][1], link_pose[1][0], link_pose[1][1])
            linearm2 = collision_class.line_obj(link_pose[1][0], link_pose[1][1], link_pose[2][0], link_pose[2][1])

            col = []
            for i in obs_list:
                col1 = collision_class.line_v_rectangle(linearm1, i)
                col.append(col1)
                col2 = collision_class.line_v_rectangle(linearm2, i)
                col.append(col2)
            
            if True in col:
                config_array_row.append(True)
            else:
                config_array_row.append(False)
        config_array.append(config_array_row)
        config_array_row = []

    grid_np = np.array(config_array).astype(int)

    return 1 - grid_np

if __name__=="__main__":
    r = planar_rr()
    obs = task_rectangle_obs_1()
    grid_np = configuration_generate_plannar_rr(r, obs)
    plt.imshow(grid_np)
    plt.show()