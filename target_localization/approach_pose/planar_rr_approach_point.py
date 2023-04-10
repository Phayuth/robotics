import os
import sys

wd = os.path.abspath(os.getcwd())
sys.path.append(str(wd))

import numpy as np
import matplotlib.pyplot as plt
from robot.planar_rr import PlanarRR
from config_space_2d.generate_config_planar_rr import configuration_generate_plannar_rr
from map.taskmap_geo_format import task_rectangle_obs_1
from map.map_value_range import map_val
from util.coord_transform import polar2cats, circle_plt

# create robot instance
robot = PlanarRR()

# SECTION - define target pose and calculate theta
x_targ = 1.6
y_targ = 2.15
target = np.array([x_targ, y_targ]).reshape(2, 1)
t_targ = robot.inverse_kinematic_geometry(target, elbow_option=0)


# SECTION - calculate approach point and calculate theta
d_app = 0.3
alpha = np.pi + np.sum(t_targ)
aprch = np.array([d_app * np.cos(alpha), d_app * np.sin(alpha)]).reshape(2, 1) + target
t_app = robot.inverse_kinematic_geometry(aprch, elbow_option=0)


# SECTION - plot task space
obs_list = task_rectangle_obs_1()
robot.plot_arm(t_app, plt_basis=True)
robot.plot_arm(t_targ)
circle_plt(x_targ, y_targ, d_app)
for obs in obs_list:
    obs.plot()
plt.show()


# SECTION - create config space
grid_np = configuration_generate_plannar_rr(robot, obs_list)


# SECTION - plot config space
theta1_app_index = int(map_val(t_app[0].item(), -np.pi, np.pi, 0, 360))
theta2_app_index = int(map_val(t_app[1].item(), -np.pi, np.pi, 0, 360))
theta1_goal_index = int(map_val(t_targ[0].item(), -np.pi, np.pi, 0, 360))
theta2_goal_index = int(map_val(t_targ[1].item(), -np.pi, np.pi, 0, 360))
grid_np[theta1_app_index, theta2_app_index] = 2
grid_np[theta1_goal_index, theta2_goal_index] = 3
plt.imshow(grid_np)
plt.show()
