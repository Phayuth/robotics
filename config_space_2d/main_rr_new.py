import os
import sys
wd = os.path.abspath(os.getcwd())
sys.path.append(str(wd))

import matplotlib.pyplot as plt
import numpy as np

from robot.planar_rr import planar_rr
from map import dummy_map
from gencofig_test import construct_config_space_2d

# Create map
map = dummy_map.map_2d_1()

# Create robot
robot = planar_rr()

# Create Configuration space
configuration = construct_config_space_2d(robot, map)

# Plot robot and obstacle in task space
plt.figure(figsize=(10,10))
plt.axes().set_aspect('equal')
theta = np.array([[np.pi/2],[0]])
plt.imshow(np.transpose(map),cmap = "gray", interpolation = 'nearest')
robot.plot_arm(theta)
plt.gca().invert_yaxis()
plt.show()

# Plot config space
plt.figure(figsize=(10,10))
plt.axes().set_aspect('equal')
plt.imshow(np.transpose(configuration),cmap = "gray", interpolation = 'nearest')
plt.show()