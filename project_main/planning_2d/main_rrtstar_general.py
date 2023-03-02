import os
import sys
wd = os.path.abspath(os.getcwd())
sys.path.append(str(wd))

import numpy as np
import math
import matplotlib.pyplot as plt
from planner.rrtstar_general import node, rrt_star
from map.generate_obstacle_space import Obstacle_generater, obstacle_generate_from_map, bmap

# Create map
map = []
map_size = np.array([0,29])

# Create start and end node
x_init = node(3, 27)
x_goal = node(27, 3)

# Create obstacle
collision_range = (2**(1/2))/2
obstacle, obstacle_center  = obstacle_generate_from_map(index=0)#bmap()
obs = Obstacle_generater(obstacle)

# Create planner
iteration = 500
m = (map_size[1]+1) * (map_size[1]+1)
r = (2 * (1 + 1/2)**(1/2)) * (m/math.pi)**(1/2)
eta =  r * (math.log(iteration) / iteration)**(1/2)
rrt = rrt_star(map, x_init, x_goal, eta, obs, obstacle_center, collision_range, map_size, iteration)

# Seed random
np.random.seed(0)

# Start planning
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

plt.plot(px,py, 'ro-',linewidth=2)
# Draw rrt tree
rrt.Draw_Tree()
plt.show()
