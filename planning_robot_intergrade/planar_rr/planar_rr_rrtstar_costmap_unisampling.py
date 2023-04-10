import os
import sys

wd = os.path.abspath(os.getcwd())
sys.path.append(str(wd))

import matplotlib.pyplot as plt
import numpy as np

from config_space_2d.generate_config_planar_rr import configuration_generate_plannar_rr
from map.map_value_range import map_val, map_multi_val
from map.taskmap_geo_format import task_rectangle_obs_1
from planner.research_rrt_2dof.rrtstar_costmap_unisampling import node, rrt_star
from robot.planar_rr import PlanarRR
from util.extract_path_class import extract_path_class_2d

# robot, inverse kinematic and plot
robot = PlanarRR()

# define task space init point and goal point
init_pose = np.array([[1.8],[3.3]])
desired_pose = np.array([[3.5],[1.8]])

# using inverse kinematic, determine the theta configuration space in continuous space
theta_init = robot.inverse_kinematic_geometry(init_pose, elbow_option=0)
theta_goal = robot.inverse_kinematic_geometry(desired_pose, elbow_option=0)

# calculate theta init index inside confuration 
theta_init_index = map_multi_val(theta_init, -np.pi, np.pi, 0, 360)

# calculate theta goal index
theta_goal_index = map_multi_val(theta_goal, -np.pi, np.pi, 0, 360)

# task space plot view
robot.plot_arm(theta_init, plt_basis=True)
robot.plot_arm(theta_goal)
obs_list = task_rectangle_obs_1()
for obs in obs_list:
    obs.plot()
plt.show()

# create config grid and view
map = configuration_generate_plannar_rr(robot, obs_list)
plt.imshow(map)
plt.show()

# Create start and end node
x_init = theta_init_index
x_goal = theta_goal_index

# Create planner
distance_weight = 0.5
obstacle_weight = 0.5
rrt = rrt_star(map, x_init, x_goal, distance_weight, obstacle_weight, maxiteration=1000)

# Seed random
np.random.seed(0)

# Start planner
rrt.start_planning()

# Get path
path = rrt.Get_Path()

# Draw rrt tree
plt.imshow(map.T)
rrt.Draw_Tree()
rrt.Draw_path(path)
plt.show()

# plot task space motion
plt.axes().set_aspect('equal')
plt.axvline(x=0, c="green")
plt.axhline(y=0, c="green")
obs_list = task_rectangle_obs_1()
for obs in obs_list:
    obs.plot()

pathx, pathy = extract_path_class_2d(path)
print("==>> pathx: \n", pathx)
print("==>> pathy: \n", pathy)

for i in range(len(path)):
    # map index image to theta
    theta1 = map_val(pathx[i], 0, 360, -np.pi, np.pi)
    theta2 = map_val(pathy[i], 0, 360, -np.pi, np.pi)
    robot.plot_arm(np.array([[theta1], [theta2]]))
    plt.pause(1)
plt.show()