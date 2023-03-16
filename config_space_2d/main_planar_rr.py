import os
import sys
wd = os.path.abspath(os.getcwd())
sys.path.append(str(wd))

import matplotlib.pyplot as plt
from map.taskmap_geo_format import task_rectangle_obs_1
from robot.planar_rr import planar_rr
import numpy as np
from map.map_value_range import map_val
from generate_config_planar_rr import configuration_generate_plannar_rr

plt.axes().set_aspect('equal')
plt.axvline(x=0, c="green")
plt.axhline(y=0, c="green")

r = planar_rr()
init_pose = np.array([[1.8],[3.3]])
desired_pose = np.array([[3.5],[1.8]])
theta_up = r.inverse_kinematic_geometry(desired_pose, elbow_option=0)
theta_down = r.inverse_kinematic_geometry(desired_pose, elbow_option=1)
r.plot_arm(theta_down)
r.plot_arm(theta_up)

theta1_up = theta_up[0,0]
theta2_up = theta_up[1,0]
theta1_up_index = int(map_val(theta1_up, -np.pi, np.pi, 0, 360)) 
theta2_up_index = int(map_val(theta2_up, -np.pi, np.pi, 0, 360))
print("==>> theta1_up_index: ", theta1_up_index)
print("==>> theta2_up_index: ", theta2_up_index)

theta1_dn = theta_down[0,0]
theta2_dn = theta_down[1,0]
theta1_dn_index = int(map_val(theta1_dn, -np.pi, np.pi, 0, 360)) 
theta2_dn_index = int(map_val(theta2_dn, -np.pi, np.pi, 0, 360))
print("==>> theta1_dn_index: ", theta1_dn_index)
print("==>> theta2_dn_index: ", theta2_dn_index)

obs_list = task_rectangle_obs_1()
for obs in obs_list:
    obs.plot()
plt.show()

grid_np = configuration_generate_plannar_rr(r, obs_list)

# add a point in index of grid
plt.scatter(theta1_up_index, theta2_up_index)
plt.scatter(theta1_dn_index, theta2_dn_index)

grid_np[theta1_up_index, theta2_up_index] = 2
grid_np[theta1_dn_index, theta2_dn_index] = 2

print(grid_np.shape)
plt.imshow(grid_np.T)
plt.grid(True)
plt.show()