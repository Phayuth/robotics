import os
import sys
wd = os.path.abspath(os.getcwd())
sys.path.append(str(wd))

import numpy as np
import math
import matplotlib.pyplot as plt

from map.generate_map import grid_map_binary
from planner.rrtstar_probabilty_proxy_collision import node , rrt_star

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
rrt = rrt_star(x_init, x_goal, map, eta, distance_weight, obstacle_weight, iteration)

# Seed random
np.random.seed(0)

# Start planner
rrt.start_planning()

# Get path
path = rrt.Get_Path()
all_path = []
for i in path:
    point = [i.x,i.y]
    all_path.append(point)

print(all_path)
px = []
py = []

for j in all_path:
    px.append(j[0])
    py.append(j[1])
plt.imshow(map.T,origin='lower')
plt.plot(px,py, 'ro-',linewidth=2)
# Draw rrt tree
rrt.Draw_Tree()
# rrt.Draw_path(path)
plt.show()