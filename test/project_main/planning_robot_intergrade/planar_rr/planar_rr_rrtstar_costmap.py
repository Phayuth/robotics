import os
import sys

wd = os.path.abspath(os.getcwd())
sys.path.append(str(wd))

import matplotlib.pyplot as plt
import numpy as np

from config_space_2d.generate_config_planar_rr import configuration_generate_plannar_rr
from map.mapclass import map_val, map_vec
from map.taskmap_geo_format import task_rectangle_obs_1
from map.mapclass import CostMapLoader, CostMapClass
from planner.rrtstar_costmap import RrtstarCostmap
from robot.planar_rr import PlanarRR
from planner.extract_path_class import extract_path_class_2d

# robot, inverse kinematic and plot
robot = PlanarRR()

# define task space init point and goal point
init_pose = np.array([[1.8],[3.3]])
desired_pose = np.array([[3.5],[1.8]])

# using inverse kinematic, determine the theta configuration space in continuous space
theta_init = robot.inverse_kinematic_geometry(init_pose, elbow_option=0)
theta_goal = robot.inverse_kinematic_geometry(desired_pose, elbow_option=0)

# calculate theta index inside confuration 
x_init = map_vec(theta_init, -np.pi, np.pi, 0, 360)
x_goal = map_vec(theta_goal, -np.pi, np.pi, 0, 360)

# task space plot view
robot.plot_arm(theta_init, plt_basis=True)
robot.plot_arm(theta_goal)
obs_list = task_rectangle_obs_1()
for obs in obs_list:
    obs.plot()
plt.show()

# create config grid and view
map = configuration_generate_plannar_rr(robot, obs_list)
maploader = CostMapLoader.loadarray(map)
mapClass = CostMapClass(maploader=maploader)
plt.imshow(map)
plt.show()

# Create planner
np.random.seed(0)
rrt = RrtstarCostmap(mapClass, x_init, x_goal, distanceWeight=0.5, obstacleWeight=0.5, maxIteration=2000)
rrt.planning()
path = rrt.get_path()
rrt.plt_env()
rrt.draw_path(path)
plt.show()

# plot task space motion
plt.axes().set_aspect('equal')
plt.axvline(x=0, c="green")
plt.axhline(y=0, c="green")
obs_list = task_rectangle_obs_1()
for obs in obs_list:
    obs.plot()

pathx, pathy = extract_path_class_2d(path)
print("==>> pathx: \n", pathx)
print("==>> pathy: \n", pathy)

for i in range(len(path)):
    # map index image to theta
    theta1 = map_val(pathx[i], 0, 360, -np.pi, np.pi)
    theta2 = map_val(pathy[i], 0, 360, -np.pi, np.pi)
    robot.plot_arm(np.array([[theta1], [theta2]]))
    plt.pause(1)
plt.show()