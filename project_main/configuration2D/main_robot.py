from RobotArm2D import Robot,map
import matplotlib.pyplot as plt
import numpy as np

map = map() # create map

base_position = [15, 15]
link_lenths = [5, 5]
robot = Robot(base_position, link_lenths, map)
configuration = robot.construct_config_space()

plt.figure(figsize=(10,10))
plt.axes().set_aspect('equal')
plt.imshow(np.transpose(configuration),cmap = "gray", interpolation = 'nearest')
plt.show()

plt.figure(figsize=(10,10))
plt.axes().set_aspect('equal')
r1 = robot.robot_position(90,0)
plt.plot([robot.base_position[0], r1[0][0]],[robot.base_position[1], r1[0][1]] , "b", linewidth=8)
plt.plot([r1[0][0], r1[1][0]],[r1[0][1], r1[1][1]] , "b", linewidth=8)
plt.plot([r1[1][0], r1[2][0]],[r1[1][1], r1[2][1]] , "r", linewidth=8)


plt.imshow(np.transpose(map),cmap = "gray", interpolation = 'nearest')
plt.gca().invert_yaxis()
plt.show()