import os
import sys
wd = os.path.abspath(os.getcwd()) # get the top parent folder
sys.path.append(str(wd)) # add it to path

import numpy as np
import math
import matplotlib.pyplot as plt
from planner.research_rrtstar.rrt_general import node, rrt_general
from map.generate_obstacle_space import Obstacle_generater, obstacle_generate_from_map

# Create obstacle in task space
map, obstacle, obstacle_center = obstacle_generate_from_map(index=0)
obs = Obstacle_generater(obstacle)
collision_range = (2**(1/2))/2

# Create start and end node
x_init = node(5, 5)
x_goal = node(20, 20)

# Create planner
iteration = 2000
m = (map.shape[0]) * (map.shape[1])
r = (2 * (1 + 1/2)**(1/2)) * (m/math.pi)**(1/2)
eta = r*(math.log(iteration) / iteration)**(1/2)
rrt = rrt_general(map, x_init, x_goal, eta, obs, obstacle_center, collision_range, iteration)

# Seed random
np.random.seed(0)

# Call start planning
rrt.start_planning()

# Get path afte planing
path = rrt.Get_Path()

# Print time statistic
rrt.print_time()
rrt.Draw_obs()
rrt.Draw_Tree()
rrt.Draw_path(path)
plt.show()
