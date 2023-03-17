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

# robot, inverse kinematic and plot
r = planar_rr()
init_pose = np.array([[1.8],[3.3]])
desired_pose = np.array([[3.5],[1.8]])
theta_up = r.inverse_kinematic_geometry(desired_pose, elbow_option=0)
r.plot_arm(theta_up)

# map theta to index image (option elbow up)
theta1_up = theta_up[0,0]
print("==>> theta1_up: ", theta1_up)
theta2_up = theta_up[1,0]
print("==>> theta2_up: ", theta2_up)
theta1_up_index = int(map_val(theta1_up, -np.pi, np.pi, 0, 360)) 
print("==>> theta1_up_index: ", theta1_up_index)
theta2_up_index = int(map_val(theta2_up, -np.pi, np.pi, 0, 360))
print("==>> theta2_up_index: ", theta2_up_index)

# map index image to theta (option elbow up)
theta1 = map_val(theta1_up_index, 0, 360, -np.pi, np.pi)
print("==>> theta1: ", theta1)
theta2 = map_val(theta2_up_index, 0, 360, -np.pi, np.pi)
print("==>> theta2: ", theta2)

obs_list = task_rectangle_obs_1()
for obs in obs_list:
    obs.plot()
plt.show()

grid_np = configuration_generate_plannar_rr(r, obs_list)

# add a point in index of grid
grid_np[theta1_up_index, theta2_up_index] = 2

plt.imshow(grid_np)
plt.grid(True)
plt.show()