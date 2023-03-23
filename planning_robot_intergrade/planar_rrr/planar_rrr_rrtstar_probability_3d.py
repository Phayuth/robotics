import os
import sys
wd = os.path.abspath(os.getcwd())
sys.path.append(str(wd))

import matplotlib.pyplot as plt
from map.taskmap_geo_format import task_rectangle_obs_6
from robot.planar_rrr import planar_rrr
import numpy as np
from map.map_value_range import map_val, map_multi_val
from config_space_2d.generate_config_planar_rrr import configuration_generate_plannar_rrr
from planner.research_rrtstar_3d.rrtstar_probabilty_3d import node , rrt_star
from util.extract_path_class import extract_path_class_3d

plt.axes().set_aspect('equal')
plt.axvline(x=0, c="green")
plt.axhline(y=0, c="green")

# robot, inverse kinematic and plot
robot = planar_rrr()

# define task space init point and goal point
init_pose = np.array([[2.5],[0],[0]])
# desired_pose = np.array([[0.9],[0.9],[0]])
desired_pose = np.array([[1.5],[1],[0]])

# using inverse kinematic, determine the theta configuration space in continuous space
theta_init = robot.inverse_kinematic_geometry(init_pose, elbow_option=0)
theta_goal = robot.inverse_kinematic_geometry(desired_pose, elbow_option=0)

# grid size
grid_size = 75
# calculate theta init index inside confuration 
theta_init_index = (map_multi_val(theta_init, -np.pi, np.pi, 0, grid_size)).astype(int)
# calculate theta goal index
theta_goal_index = (map_multi_val(theta_goal, -np.pi, np.pi, 0, grid_size)).astype(int)


# task space plot view
robot.plot_arm(theta_init)
robot.plot_arm(theta_goal)
obs_list = task_rectangle_obs_6()
for obs in obs_list:
    obs.plot()
plt.show()

# create config grid and view
map = configuration_generate_plannar_rrr(robot, obs_list)
# np.save('config_rrr.npy', map)
# map = np.load('.\map\mapdata\config_rrr.npy')

# Planing
x_init = node(theta_init_index[0,0], theta_init_index[1,0], theta_init_index[2,0])
x_goal = node(theta_goal_index[0,0], theta_goal_index[1,0], theta_goal_index[2,0])
iteration = 1000
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

# plot task space motion
plt.axes().set_aspect('equal')
plt.axvline(x=0, c="green")
plt.axhline(y=0, c="green")
obs_list = task_rectangle_obs_6()
for obs in obs_list:
    obs.plot()

pathx, pathy, pathz = extract_path_class_3d(path)
print("==>> pathx: ", pathx)
print("==>> pathy: ", pathy)
print("==>> pathz: ", pathz)

for i in range(len(path)):
    # map index image to theta
    theta1 = map_val(pathx[i], 0, grid_size, -np.pi, np.pi)
    theta2 = map_val(pathy[i], 0, grid_size, -np.pi, np.pi)
    theta3 = map_val(pathz[i], 0, grid_size, -np.pi, np.pi)
    robot.plot_arm(np.array([[theta1], [theta2], [theta3]]))
    plt.pause(1)
plt.show()