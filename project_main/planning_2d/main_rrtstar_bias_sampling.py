import os
import sys
wd = os.path.abspath(os.getcwd())
sys.path.append(str(wd))

import numpy as np
import math
import matplotlib.pyplot as plt
from planner.rrtstar_probabilty_bias_sampling import node, rrt_star
from map.generate_obstacle_space import Obstacle_generater, obstacle_generate_from_map
from map.generate_map import grid_map_binary

# Create map
index = 1
map = grid_map_binary(index)
plt.imshow(map)
plt.show()

# Get obstacle from map
mm, obstacle, obstacle_center  = obstacle_generate_from_map(index)
obs = Obstacle_generater(obstacle)
collision_range = (2**(1/2))/2

# Create start and stop
x_init = node(15, 15)
x_goal = node(20, 20)

# Create planner
map_size = map.shape
iteration = 1000
m = (map_size[1]+1) * (map_size[1]+1)
r = (2 * (1 + 1/2)**(1/2)) * (m/math.pi)**(1/2)
eta =  r * (math.log(iteration) / iteration)**(1/2)
rrt = rrt_star(map, x_init, x_goal, eta, obs, obstacle_center, collision_range, iteration)

# Start planner
rrt.start_planning()

# Get path result
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

plt.plot(px,py, 'ro-',linewidth=2)
# Draw rrt tree
rrt.Draw_Tree()
rrt.Draw_obs()
plt.show()
