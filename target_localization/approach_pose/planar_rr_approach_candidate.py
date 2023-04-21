import os
import sys

wd = os.path.abspath(os.getcwd())
sys.path.append(str(wd))

import numpy as np
import matplotlib.pyplot as plt
from robot.planar_rr import PlanarRR
from config_space_2d.generate_config_planar_rr import configuration_generate_plannar_rr
from map.taskmap_geo_format import task_rectangle_obs_1
from map.mapclass import map_val
from util.coord_transform import polar2cats

# create robot instance
robot = PlanarRR()

# SECTION - user define pose
x_targ = 1.5
y_targ = 1.5
t_targ = np.linspace(np.pi / 2, 3 * np.pi / 2, 10)  # candidate pose
app_d = 0.5
app_x, app_y = polar2cats(app_d, t_targ, x_targ, y_targ)
obs_list = task_rectangle_obs_1()


# SECTION - calculate ik for each approach pose and the main pose
t_app = [robot.inverse_kinematic_geometry(np.array([[app_x[i]], [app_y[i]]]), elbow_option=0) for i in range(len(app_x))]
t_app_main = robot.inverse_kinematic_geometry(np.array([[x_targ], [y_targ]]), elbow_option=0)


# SECTION - create config space
grid_np = configuration_generate_plannar_rr(robot, obs_list)


# SECTION - plot task space
plt.axes().set_aspect('equal')
plt.axvline(x=0, c="black")
plt.axhline(y=0, c="black")
plt.plot(app_x, app_y, c="green")
for obs in obs_list:
    obs.plot()
for each_theta in range(len(t_app)):
    robot.plot_arm(np.array([[t_app[each_theta][0].item()], [t_app[each_theta][1].item()]]))
plt.show()


# SECTION - plot configspace pose
theta1_plt = []
theta2_plt = []
for k in range(len(t_app)):
    theta1_index = int(map_val(t_app[k][0].item(), -np.pi, np.pi, 0, 360))
    theta2_index = int(map_val(t_app[k][1].item(), -np.pi, np.pi, 0, 360))
    theta1_plt.append(theta1_index)
    theta2_plt.append(theta2_index)
theta1_main_plt = int(map_val(t_app_main[0].item(), -np.pi, np.pi, 0, 360))
theta2_main_plt = int(map_val(t_app_main[1].item(), -np.pi, np.pi, 0, 360))

for u in range(len(theta1_plt)):
    grid_np[theta2_plt[u], theta1_plt[u]] = 2
grid_np[theta2_main_plt, theta1_main_plt] = 3
plt.imshow(grid_np)
plt.show()