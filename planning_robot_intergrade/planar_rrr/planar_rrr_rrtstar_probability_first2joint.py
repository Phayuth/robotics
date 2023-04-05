import os
import sys

wd = os.path.abspath(os.getcwd())
sys.path.append(str(wd))

import matplotlib.pyplot as plt
import numpy as np

from config_space_2d.generate_config_planar_rrr import configuration_generate_plannar_rrr_first_2joints
from map.map_value_range import map_multi_val, map_val
from map.taskmap_geo_format import task_rectangle_obs_6
from planner.research_rrtstar_3d.rrtstar_probabilty_3d import node, rrt_star
from robot.planar_rrr import planar_rrr
from util.coord_transform import circle_plt, polar2cats
from util.extract_path_class import extract_path_class_3d

robot = planar_rrr()

# define task space init point and goal point
init_pose = np.array([[2.5],[0],[0]])
x_targ = 0.5
y_targ = 0.5
alpha_candidate = 2  # given from grapse pose candidate

# target
phi_target = alpha_candidate - np.pi
target = np.array([[x_targ],
                   [y_targ],
                   [phi_target]])
theta_ik_tag = robot.inverse_kinematic_geometry(target, elbow_option=0)


# approach point
d_app = 0.1
app_point = np.array([[d_app*np.cos(target[2, 0]+np.pi) + target[0, 0]],
                      [d_app*np.sin(target[2, 0]+np.pi) + target[1, 0]],
                      [target[2, 0]]])
theta_ik_app = robot.inverse_kinematic_geometry(app_point, elbow_option=0)


# plot view
robot.plot_arm(theta_ik_tag, plt_basis=True)
robot.plot_arm(theta_ik_app)
circle_plt(x_targ, y_targ, radius=0.1)
obs_list = task_rectangle_obs_6()
for obs in obs_list:
    obs.plot()
plt.show()


# using inverse kinematic, determine the theta configuration space in continuous space
theta_init = robot.inverse_kinematic_geometry(init_pose, elbow_option=0)
theta_goal = theta_ik_app

# grid size
grid_size = 75
theta_init_index = (map_multi_val(theta_init, -np.pi, np.pi, 0, grid_size)).astype(int)
theta_goal_index = (map_multi_val(theta_goal, -np.pi, np.pi, 0, grid_size)).astype(int)

map = configuration_generate_plannar_rrr_first_2joints(robot, obs_list)

# Planing
x_init = np.array([theta_init_index[0,0], theta_init_index[1,0], theta_init_index[2,0]]).reshape(3,1)
x_goal = np.array([theta_goal_index[0,0], theta_goal_index[1,0], theta_goal_index[2,0]]).reshape(3,1)
iteration = 500
m = map.shape[0] * map.shape[1] * map.shape[2]
r = (2 * (1 + 1/2)**(1/2)) * (m/np.pi)**(1/2)
eta = r * (np.log(iteration) / iteration)**(1/2)
distance_weight = 0.5
obstacle_weight = 0.5
rrt = rrt_star(map, x_init, x_goal, eta, distance_weight, obstacle_weight, iteration)
np.random.seed(0)
rrt.start_planning()
path = rrt.Get_Path()

# Draw rrt tree
rrt.Draw_tree()
rrt.Draw_path(path)
plt.show()

pathx, pathy, pathz = extract_path_class_3d(path)


# plot view
circle_plt(x_targ, y_targ, radius=0.1)
for i in range(len(path)):
    theta1 = map_val(pathx[i], 0, grid_size, -np.pi, np.pi)
    theta2 = map_val(pathy[i], 0, grid_size, -np.pi, np.pi)
    theta3 = map_val(pathz[i], 0, grid_size, -np.pi, np.pi)
    robot.plot_arm(np.array([[theta1], [theta2], [theta3]]))
    plt.pause(1)
plt.show()