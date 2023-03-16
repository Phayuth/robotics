import os
import sys
wd = os.path.abspath(os.getcwd())
sys.path.append(str(wd))

import matplotlib.pyplot as plt
from collision_check_geometry import collision_class
from map.taskmap_geo_format import task_rectangle_obs
from robot.planar_rr import planar_rr
import numpy as np

r = planar_rr()

obs = task_rectangle_obs()

theta1 = np.linspace(-np.pi, np.pi, 360)
theta2 = np.linspace(-np.pi, np.pi, 360)

config_array = []
config_array_row = []

for j in theta1:
    for k in theta2:
        print(f"at theta1 {j} | at theta2 {k}")
        theta = np.array([[j],[k]])
        link_pose = r.forward_kinematic(theta, return_link_pos=True)
        linearm1 = collision_class.line_obj(link_pose[0][0], link_pose[0][1], link_pose[1][0], link_pose[1][1])
        linearm2 = collision_class.line_obj(link_pose[1][0], link_pose[1][1], link_pose[2][0], link_pose[2][1])

        col = []
        for i in obs:
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
print(grid_np.shape)
plt.imshow(grid_np)
plt.grid(True)
plt.show()