import os
import sys
wd = os.path.abspath(os.getcwd())
sys.path.append(str(wd))

import numpy as np
import math
import matplotlib.pyplot as plt
from planner.research_rrtstar.rrtstar_general import node, rrt_star
from map.generate_obstacle_space import Obstacle_generater, obstacle_generate_from_map, bmap

# Map and Create obstacle
map, obstacle, obstacle_center = obstacle_generate_from_map(index=0)
obs = Obstacle_generater(obstacle)
collision_range = (2**(1/2))/2

# Create start and end node
x_init = node(3, 27)
x_goal = node(27, 3)

# Create planner
iteration = 500
m = (map.shape[0]) * (map.shape[1])
r = (2 * (1 + 1/2)**(1/2)) * (m/math.pi)**(1/2)
eta = r*(math.log(iteration) / iteration)**(1/2)
rrt = rrt_star(map, x_init, x_goal, eta, obs, obstacle_center, collision_range, iteration)

# Seed random
np.random.seed(0)

# Start planning
rrt.start_planning()

# Get path
path = rrt.Get_Path()

# Draw rrt tree
rrt.Draw_obs()
rrt.Draw_Tree()
rrt.Draw_path(path)
plt.show()
