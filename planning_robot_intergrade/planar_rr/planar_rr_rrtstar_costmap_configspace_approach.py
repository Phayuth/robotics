import os
import sys

wd = os.path.abspath(os.getcwd())
sys.path.append(str(wd))

import matplotlib.pyplot as plt
import numpy as np

from config_space_2d.generate_config_planar_rr import configuration_generate_plannar_rr
from map.mapclass import map_val, map_vec
from map.taskmap_geo_format import task_rectangle_obs_5
from map.mapclass import CostMapLoader, CostMapClass
from planner.rrtstar_costmap_configspace import RrtstarCostmapConfigspace
from robot.planar_rr import PlanarRR
from util.coord_transform import circle_plt, polar2cats
from util.extract_path_class import extract_path_class_2d

# robot, inverse kinematic and plot
robot = PlanarRR()

# init pose
init_pose = np.array([[4],[0]])
x_init = robot.inverse_kinematic_geometry(init_pose, elbow_option=0)

# goal pose
x_targ = 1.6
y_targ = 2.15
goal_pose = np.array([[1.6],[2.15]])
x_goal = robot.inverse_kinematic_geometry(goal_pose, elbow_option=0)

# approach pose
d_app = 0.1
alpha = np.pi + np.sum(x_goal)
app_point = np.array([[d_app*np.cos(alpha)],
                      [d_app*np.sin(alpha)]]) + goal_pose
x_appr = robot.inverse_kinematic_geometry(app_point, elbow_option=0)

grid_size = 360

# task space plot view
robot.plot_arm(x_init, plt_basis=True)
robot.plot_arm(x_goal)
robot.plot_arm(x_appr)
circle_plt(x_targ, y_targ, d_app)
obs_list = task_rectangle_obs_5()
for obs in obs_list:
    obs.plot()
plt.show()

# create config grid and view
map = configuration_generate_plannar_rr(robot, obs_list)
maploader = CostMapLoader.loadarray(map)
mapClass = CostMapClass(maploader=maploader, maprange=[[-np.pi, np.pi], [-np.pi, np.pi]])
plt.imshow(map)
plt.show()

# Planning
np.random.seed(0)
rrt = RrtstarCostmapConfigspace(mapClass, x_init, x_appr, distanceWeight= 0.5, obstacleWeight= 0.5, maxIteration=1000)
rrt.planning()
path = rrt.get_path()
rrt.plt_env()
rrt.draw_path(path)
plt.show()

# plot task space motion
plt.axes().set_aspect('equal')
plt.axvline(x=0, c="green")
plt.axhline(y=0, c="green")
obs_list = task_rectangle_obs_5()
for obs in obs_list:
    obs.plot()

pathx, pathy = extract_path_class_2d(path)

# plot real target
plt.scatter(goal_pose[0,0], goal_pose[1,0])
circle_plt(x_targ, y_targ, d_app)
robot.plot_arm(x_goal)
for i in range(len(path)):
    robot.plot_arm(np.array([[pathx[i]], [pathy[i]]]))
    plt.pause(1)
plt.show()