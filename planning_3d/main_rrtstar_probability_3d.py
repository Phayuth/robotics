import os
import sys
wd = os.path.abspath(os.getcwd())
sys.path.append(str(wd))

from planner.research_rrtstar.rrtstar_probabilty_3d import node, rrt_star
# from map.load_map import map_3d, Reshape_map
import math
import numpy as np

import matplotlib.pyplot as plt

# Creat map
filter_size = 0  # 1 = 3x3, 2 = 5x5, 3 = 7x7
classify = False
# map = map_3d()
map = np.load('./map/mapdata/config_space_data_3d/config3D.npy')

# Create Start and end node
x_init = node(0, 0, 0)
x_goal = node(40, 10, 30)

# Create planner
iteration = 1000
m = map.shape[0] * map.shape[1] * map.shape[2]
r = (2 * (1 + 1/2)**(1/2)) * (m/math.pi)**(1/2)
eta = r * (math.log(iteration) / iteration)**(1/2)
distance_weight = 0.5
obstacle_weight = 0.5
rrt = rrt_star(map, x_init, x_goal, eta, distance_weight, obstacle_weight, iteration)

# Seed random
np.random.seed(0)

# Call to start planning
rrt.start_planning()

# Get path from planner
path = rrt.Get_Path()

# Print time
rrt.print_time()
rrt.Draw_tree()
rrt.Draw_path(path)
plt.show()