import os
import sys
wd = os.path.abspath(os.getcwd())
sys.path.append(str(wd))

import numpy as np
import math
import matplotlib.pyplot as plt

from map.load_map import grid_map_binary
from planner.research_rrtstar.rrtstar_probabilty_proxy_collision import node , rrt_star

# Create map
map = grid_map_binary(index=1).T
plt.imshow(map)
plt.show()

# Create start and end node
x_init = node(24, 12)
x_goal = node(27, 27)

# Create planner
iteration = 1000
m = map.shape[0] * map.shape[1]
r = (2 * (1 + 1/2)**(1/2)) * (m/math.pi)**(1/2)
eta =  r * (math.log(iteration) / iteration)**(1/2)
distance_weight = 0.5
obstacle_weight = 0.5
rrt = rrt_star(map, x_init, x_goal, eta, distance_weight, obstacle_weight, iteration)

# Seed random
np.random.seed(0)

# Start planner
rrt.start_planning()

# Get path
path = rrt.Get_Path()

# Draw rrt tree
plt.imshow(map.T,origin='lower')
rrt.Draw_Tree()
rrt.Draw_path(path)
plt.show()