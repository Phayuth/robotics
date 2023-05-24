import os
import sys
wd = os.path.abspath(os.getcwd())
sys.path.append(str(wd))

from planner.research_rrtstar_3d.rrtstar_costmap_biassampling import node, rrt_star
from map.taskmap_img_format import map_3d_empty
import math
import numpy as np
from planner_util.extract_path_class import extract_path_class_3d
import matplotlib.pyplot as plt

# Creat map
filter_size = 0  # 1 = 3x3, 2 = 5x5, 3 = 7x7
classify = False
map = map_3d_empty()
# map = np.load('./map/mapdata/config_space_data_3d/config3D.npy')

# Create Start and end node
x_init = node(0, 0, 0)
x_goal = node(40, 10, 30)

# Create planner
iteration = 1000
m = map.shape[0] * map.shape[1] * map.shape[2]
r = (2 * (1 + 1/2)**(1/2)) * (m/math.pi)**(1/2)
eta = r * (math.log(iteration) / iteration)**(1/2)
eta = 10
distance_weight = 0.5
obstacle_weight = 0.5
rrt = rrt_star(map, x_init, x_goal, eta, distance_weight, obstacle_weight, iteration)

# Seed random
np.random.seed(0)

# Call to start planning
rrt.start_planning()

# Get path from planner
path = rrt.Get_Path()
print(len(path))
pathx, pathy, pathz= extract_path_class_3d(path)
print("==>> pathx: \n", pathx)
print("==>> pathy: \n", pathy)
print("==>> pathz: \n", pathz)

# Print time
rrt.print_time()
rrt.Draw_tree()
rrt.Draw_path(path)
plt.show()