import os
import sys
wd = os.path.abspath(os.getcwd())
sys.path.append(str(wd))

import numpy as np
from collision_check_geometry import collision_class


def configuration_generate_plannar_rr(robot, obs_list):

    grid_size = 360
    theta1 = np.linspace(-np.pi, np.pi, grid_size)
    theta2 = np.linspace(-np.pi, np.pi, grid_size)

    grid_map = np.zeros((grid_size, grid_size))

    for j in range(len(theta1)):
        for k in range(len(theta2)):
            print(f"at theta1 {theta1[j]} | at theta2 {theta2[k]}")
            theta = np.array([[theta1[j]],[theta2[k]]])
            link_pose = robot.forward_kinematic(theta, return_link_pos=True)
            linearm1 = collision_class.obj_line2d(link_pose[0][0], link_pose[0][1], link_pose[1][0], link_pose[1][1])
            linearm2 = collision_class.obj_line2d(link_pose[1][0], link_pose[1][1], link_pose[2][0], link_pose[2][1])

            col = []
            for i in obs_list:
                col1 = collision_class.intersect_line_v_rectangle(linearm1, i)
                col2 = collision_class.intersect_line_v_rectangle(linearm2, i)
                col.extend((col1,col2))

            if True in col:
                grid_map[j,k] = 1

    return 1 - grid_map

if __name__=="__main__":
    import matplotlib.pyplot as plt
    from map.taskmap_geo_format import task_rectangle_obs_1
    from robot.planar_rr import planar_rr
    from map.map_value_range import map_multi_val


    # SECTION - robot class 
    r = planar_rr()
    init_pose = np.array([[1.8],[3.3]])
    desired_pose = np.array([[3.5],[1.8]])
    theta = r.inverse_kinematic_geometry(desired_pose, elbow_option=0)
    theta_index = map_multi_val(theta, -np.pi, np.pi, 0, 360).astype(int) # map theta to index image (option elbow up)
    theta_val = map_multi_val(theta_index, 0, 360, -np.pi, np.pi)         # map index image to theta (option elbow up)
    obs_list = task_rectangle_obs_1()


    # SECTION - plot task space
    r.plot_arm(theta, plt_basis=True)
    for obs in obs_list:
        obs.plot()
    plt.show()


    # SECTION - configuration space
    grid_np = configuration_generate_plannar_rr(r, obs_list)
    grid_np[theta_index[0,0], theta_index[1,0]] = 2 # add a point in index of grid


    # SECTION - plot configspace in image format
    plt.imshow(grid_np)
    plt.grid(True)
    plt.show()