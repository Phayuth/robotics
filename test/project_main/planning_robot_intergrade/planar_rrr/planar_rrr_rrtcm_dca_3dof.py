""" Planning path with RRTstar Costmap
Planner : RRTStar Costmap
Approch Angle : Distance Circle Approach
DOF : 3
"""
import os
import sys

wd = os.path.abspath(os.getcwd())
sys.path.append(str(wd))

import matplotlib.pyplot as plt
import numpy as np

from config_space_2d.generate_config_planar_rrr import configuration_generate_plannar_rrr
from map.mapclass import map_val, map_vec
from map.taskmap_geo_format import task_rectangle_obs_6
from planner.ready.rrt_2D.rrtstar_costmap_biassampling3d import node, rrt_star
from robot.planar_rrr import PlanarRRR
from planner_util.coord_transform import circle_plt, polar2cats
from planner.extract_path_class import extract_path_class_3d

# robot, inverse kinematic and plot
robot = PlanarRRR()

# define task space init point and goal point
x_targ = 0.5
y_targ = 0.5
alpha_targ = 2  # given from grapse pose candidate
d_app = 0.1
phi_targ = alpha_targ - np.pi
target = np.array([x_targ, y_targ, phi_targ]).reshape(3, 1)
t_targ = robot.inverse_kinematic_geometry(target, elbow_option=0)

aprch = np.array([[d_app * np.cos(target[2, 0] + np.pi) + target[0, 0]], [d_app * np.sin(target[2, 0] + np.pi) + target[1, 0]], [target[2, 0]]])
theta_goal = robot.inverse_kinematic_geometry(aprch, elbow_option=0)

init_pose = np.array([[2.5],[0],[0]])
theta_init = robot.inverse_kinematic_geometry(init_pose, elbow_option=0)

# grid size
grid_size = 75
# calculate theta init index inside confuration 
x_init = map_vec(theta_init, -np.pi, np.pi, 0, grid_size)
x_goal = map_vec(theta_goal, -np.pi, np.pi, 0, grid_size)


# task space plot view
robot.plot_arm(theta_init, plt_basis=True)
robot.plot_arm(theta_goal)
robot.plot_arm(t_targ)
circle_plt(x_targ, y_targ, d_app)
# obs_list = task_rectangle_obs_6()
# for obs in obs_list:
#     obs.plot()
obs_list = []
plt.show()

# create config grid and view
map = configuration_generate_plannar_rrr(robot, obs_list)
# np.save('./map/mapdata/config_rrr.npy', map)
# map = np.load('./map/mapdata/config_rrr.npy')

# Planing
rrt = rrt_star(map, x_init, x_goal, w1=0.5, w2=0.5, maxiteration=1000)
np.random.seed(0)
rrt.start_planning()
path = rrt.Get_Path()

# Draw rrt tree
rrt.Draw_tree()
rrt.Draw_path(path)
plt.show()

# plot task space motion
plt.axes().set_aspect('equal')
plt.axvline(x=0, c="green")
plt.axhline(y=0, c="green")
circle_plt(x_targ, y_targ, d_app)
# obs_list = task_rectangle_obs_6()
# for obs in obs_list:
#     obs.plot()

pathx, pathy, pathz = extract_path_class_3d(path)
print("==>> pathx: ", pathx)
print("==>> pathy: ", pathy)
print("==>> pathz: ", pathz)

for i in range(len(path)):
    # map index image to theta
    theta1 = map_val(pathx[i], 0, grid_size, -np.pi, np.pi)
    theta2 = map_val(pathy[i], 0, grid_size, -np.pi, np.pi)
    theta3 = map_val(pathz[i], 0, grid_size, -np.pi, np.pi)
    robot.plot_arm(np.array([[theta1], [theta2], [theta3]]))
    plt.pause(1)
plt.show()