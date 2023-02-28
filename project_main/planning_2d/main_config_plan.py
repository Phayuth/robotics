import os
import sys
wd = os.path.abspath(os.getcwd())
sys.path.append(str(wd))

import numpy as np
import math
import matplotlib.pyplot as plt
from planner.rrtstar_probabilty_2d import node, rrt_star
from robot.plannar_rr.RobotArm2D import Robot
from map.generate_map import pmap
from config_space_2d.generate_config_space import construct_config_space_2d

# Load probability task map = pmap is probability map
map = pmap() #map()

# Load Robot
base_position = [15, 15]
link_lenths = [5, 5]
robot = Robot(base_position, link_lenths)

# Create probability config map
c_map = construct_config_space_2d(robot, map)

# Create start and goal node
x_init = node(35, 30)
x_goal = node(250, 300)

# Create planner
iteration = 1000
m = c_map.shape[0] * c_map.shape[1]
r = (2 * (1 + 1/2)**(1/2)) * (m/math.pi)**(1/2)
eta =  r * (math.log(iteration) / iteration)**(1/2)
distance_weight = 0.5
obstacle_weight = 0.5

rrt = rrt_star(x_init, x_goal, c_map, eta, distance_weight, obstacle_weight, iteration)

# Seed random value
np.random.seed(0)

# Call start planning
rrt.start_planning()

# Get path from planner
path = rrt.Get_Path()

# Print time
rrt.print_time()

# Plot configuration space and planner
plt.figure(figsize=(12,10))
plt.axes().set_aspect('equal')
rrt.Draw_Tree() # call to draw tree structure
plt.imshow(np.transpose(c_map), cmap = "gray", interpolation = 'nearest')
rrt.Draw_path(path)
plt.colorbar()
plt.show()

# Plot task space path
plt.figure(figsize=(10,10))
plt.axes().set_aspect('equal')
plt.imshow(np.transpose(map),cmap = "gray", interpolation = 'nearest')

w_path = []
for i in path:
    position = robot.robot_position(i.arr[0], i.arr[1])
    w_path.append(position)
for i in range(len(w_path)):
    plt.plot([w_path[i][0][0], w_path[i][1][0]], [w_path[i][0][1], w_path[i][1][1]], "k", linewidth=2.5)
    plt.plot([w_path[i][1][0], w_path[i][2][0]], [w_path[i][1][1], w_path[i][2][1]], "k", linewidth=2.5)
for i in range(len(w_path) - 1):
    plt.plot([w_path[i][2][0], w_path[i+1][2][0]], [w_path[i][2][1], w_path[i+1][2][1]], "r", linewidth=2.5)

plt.colorbar()
plt.show()