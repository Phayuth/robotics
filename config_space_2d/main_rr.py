import os
import sys
wd = os.path.abspath(os.getcwd())
sys.path.append(str(wd))

import matplotlib.pyplot as plt
import numpy as np

from robot_used.plannar_rr import RobotArm2D
from map import taskmap_img_format
from generate_config_space import construct_config_space_2d

# Create map
map = taskmap_img_format.map_2d_1()

# Create robot
base_position = [15, 15]
link_lenths = [5, 5]
robot = RobotArm2D.Robot(base_position, link_lenths)

# Create Configuration space
configuration = construct_config_space_2d(robot, map)

# Plot robot and obstacle in task space
plt.figure(figsize=(10,10))
plt.axes().set_aspect('equal')
r1 = robot.robot_position(90,0)
plt.plot([robot.base_position[0], r1[0][0]],[robot.base_position[1], r1[0][1]] , "b", linewidth=8)
plt.plot([r1[0][0], r1[1][0]],[r1[0][1], r1[1][1]] , "b", linewidth=8)
plt.plot([r1[1][0], r1[2][0]],[r1[1][1], r1[2][1]] , "r", linewidth=8)
plt.imshow(np.transpose(map),cmap = "gray", interpolation = 'nearest')
plt.gca().invert_yaxis()
plt.show()

# Plot config space
plt.figure(figsize=(10,10))
plt.axes().set_aspect('equal')
plt.imshow(np.transpose(configuration),cmap = "gray", interpolation = 'nearest')
plt.show()