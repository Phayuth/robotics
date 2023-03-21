import os
import sys
wd = os.path.abspath(os.getcwd())
sys.path.append(str(wd))

import matplotlib.pyplot as plt
from map.taskmap_geo_format import task_rectangle_obs_5
from robot.planar_rr import planar_rr
import numpy as np
from map.map_value_range import map_val
from config_space_2d.generate_config_planar_rr import configuration_generate_plannar_rr
from planner.research_rrtstar.rrtstar_probabilty_2d import node , rrt_star
from util.extract_path_class import extract_path_class_2d

# robot, inverse kinematic and plot
robot = planar_rr()

# define task space init point and goal point
# init_pose = np.array([[1.8],[3.3]])
# desired_pose = np.array([[3.5],[1.8]])

# case obstacle around target
# init_pose = np.array([[1.91],[1.27]])
desired_pose = np.array([[4],[0]])
init_pose = np.array([[1.6],[2.15]])


# using inverse kinematic, determine the theta configuration space in continuous space
theta_init = robot.inverse_kinematic_geometry(init_pose, elbow_option=0)
theta_goal = robot.inverse_kinematic_geometry(desired_pose, elbow_option=0)

# calculate theta init index inside confuration 
theta1_init = theta_init[0,0]
theta2_init = theta_init[1,0]
theta1_init_index = int(map_val(theta1_init, -np.pi, np.pi, 0, 360)) 
theta2_init_index = int(map_val(theta2_init, -np.pi, np.pi, 0, 360))

# calculate theta goal index
theta1_goal = theta_goal[0,0]
theta2_goal = theta_goal[1,0]
theta1_goal_index = int(map_val(theta1_goal, -np.pi, np.pi, 0, 360)) 
theta2_goal_index = int(map_val(theta2_goal, -np.pi, np.pi, 0, 360))

# task space plot view
robot.plot_arm(theta_init, plt_basis=True)
robot.plot_arm(theta_goal)
obs_list = task_rectangle_obs_5()
for obs in obs_list:
    obs.plot()
plt.show()

# create config grid and view
map = configuration_generate_plannar_rr(robot, obs_list)
plt.imshow(map)
plt.show()

# Create start and end node
x_init = node(theta1_init_index, theta2_init_index)
x_goal = node(theta1_goal_index, theta2_goal_index)

# Create planner
iteration = 1000
distance_weight = 0.5
obstacle_weight = 0.5
m = map.shape[0] * map.shape[1]
r = (2 * (1 + 1/2)**(1/2)) * (m/np.pi)**(1/2)
eta = r * (np.log(iteration) / iteration)**(1/2)
rrt = rrt_star(map, x_init, x_goal, eta, distance_weight, obstacle_weight, iteration)

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
obs_list = task_rectangle_obs_5()
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