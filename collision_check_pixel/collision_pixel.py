import os
import sys
wd = os.path.abspath(os.getcwd())
sys.path.append(str(wd))

import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import profile_line
from robot_used.plannar_rr.RobotArm2D import Robot
from map.taskmap_img_format import pmap, bmap

def construct_config_space_2d(robot, map, grid = 361):

    configuration_space = []
    theta1, theta2 = np.linspace(0, 360, grid), np.linspace(0, 360, grid)

    for i in theta1:
        for j in theta2:

            robot_position = robot.robot_position(i, j)

            prob = 0
            profile_prob1 = profile_line(map, robot_position[0], robot_position[1], linewidth=2, order=0, reduce_func=None)
            profile_prob2 = profile_line(map, robot_position[1], robot_position[2], linewidth=2, order=0, reduce_func=None)
            profile_prob = np.concatenate((profile_prob1, profile_prob2))
            
            if 0 in profile_prob:
                prob = 0
            else:
                prob = np.min(profile_prob)

            configuration_space.append([i, j, prob])

            print(f"At theta 1: {i} | At theta 2: {j}")

    c_map = np.zeros((361,361))
    for i in range(361):
        for j in range(361):
            c_map[i][j] = configuration_space[i*361+j][2]

    return c_map

map = bmap() #map()

# Load Robot
base_position = [15, 15]
link_lenths = [5, 5]
robot = Robot(base_position, link_lenths)

# Create probability config map
c_map = construct_config_space_2d(robot, map)
plt.imshow(c_map)
plt.show()