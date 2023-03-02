import os
import sys
wd = os.path.abspath(os.getcwd()) # get the top parent folder
sys.path.append(str(wd)) # add it to path

import numpy as np
import math
import matplotlib.pyplot as plt
from planner.rrt_general import node, rrt_general
from map.generate_obstacle_space import Obstacle_generater, obstacle_generate_from_map

# Create map
map_size = np.array([0,49])

# Create obstacle in task space
collision_range = (2**(1/2))/2
obstacle, obstacle_center  = obstacle_generate_from_map(index=1)
obs = Obstacle_generater(obstacle)

# Create start and end node
x_init = node(5, 5)
x_goal = node(20, 20)

# Create planner
iteration = 2000
m = (map_size[1]+1) * (map_size[1]+1)
r = (2 * (1 + 1/2)**(1/2)) * (m/math.pi)**(1/2)
eta = r*(math.log(iteration) / iteration)**(1/2)
sample_taken = 0
total_iter = 0
rrt = rrt_general(map, x_init, x_goal, eta, obs, obstacle_center, collision_range, map_size, iteration)

# Seed random
np.random.seed(0)

# Call start planning
rrt.start_planning()

# Get path afte planing
path = rrt.Get_Path()
print(path)

# Print time statistic
rrt.print_time()

rrt.Draw_Tree()
plt.show()