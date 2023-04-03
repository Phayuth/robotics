import os
import sys
wd = os.path.abspath(os.getcwd())
sys.path.append(str(wd))

import numpy as np
import matplotlib.pyplot as plt
from robot.planar_rr import planar_rr
from config_space_2d.generate_config_planar_rr import configuration_generate_plannar_rr
from map.taskmap_geo_format import task_rectangle_obs_1
from map.map_value_range import map_val
from util.coord_transform import polar2cats, approach_circle_plt

robot = planar_rr()

# target
x_targ = 1.6
y_targ = 2.15
target = np.array([[x_targ],
                   [y_targ]])
theta_ik_tag = robot.inverse_kinematic_geometry(target, elbow_option=0)

# approach point
d_app = 0.3
alpha = np.pi + np.sum(theta_ik_tag)
app_point = np.array([[d_app*np.cos(alpha)],
                      [d_app*np.sin(alpha)]]) + target
theta_ik_app = robot.inverse_kinematic_geometry(app_point, elbow_option=0)

# import obs list
obs_list = task_rectangle_obs_1()

# setup plot look
robot.plot_arm(theta_ik_app, plt_basis=True)
robot.plot_arm(theta_ik_tag)
approach_circle_plt(x_targ, y_targ, d_app)
for obs in obs_list:
    obs.plot()
plt.show()

# create config space
grid_np = configuration_generate_plannar_rr(robot, obs_list)

theta1_app_index = int(map_val(theta_ik_app[0].item(), -np.pi, np.pi, 0, 360)) 
theta2_app_index = int(map_val(theta_ik_app[1].item(), -np.pi, np.pi, 0, 360))
theta1_goal_index = int(map_val(theta_ik_tag[0].item(), -np.pi, np.pi, 0, 360)) 
theta2_goal_index = int(map_val(theta_ik_tag[1].item(), -np.pi, np.pi, 0, 360))

grid_np[theta1_app_index, theta2_app_index] = 2
grid_np[theta1_goal_index, theta2_goal_index] = 3

plt.imshow(grid_np)
plt.show()
