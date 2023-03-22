import os
import sys
wd = os.path.abspath(os.getcwd())
sys.path.append(str(wd))

import matplotlib.pyplot as plt
from map.taskmap_geo_format import task_rectangle_obs_5
from robot.planar_rr import planar_rr
import numpy as np
from map.map_value_range import map_val, map_multi_val
from config_space_2d.generate_config_planar_rr import configuration_generate_plannar_rr
from planner.research_rrtstar.rrtstar_probabilty_2d import node , rrt_star
from util.extract_path_class import extract_path_class_2d

def polar2cats(r,theta):
    x = r*np.cos(theta) + 1.6
    y = r*np.sin(theta) + 2.15
    return x,y
theta_view = np.linspace(np.pi/2,3*np.pi/2,10)
radius = 0.1
x_coord, y_coord = polar2cats(radius,theta_view)

# robot, inverse kinematic and plot
robot = planar_rr()

# init pose
init_pose = np.array([[4],[0]])
theta_init = robot.inverse_kinematic_geometry(init_pose, elbow_option=0)

# goal pose
goal_pose = np.array([[1.6],[2.15]])
theta_goal = robot.inverse_kinematic_geometry(goal_pose, elbow_option=0)

# approach pose
d_app = 0.1
alpha = np.pi + np.sum(theta_goal)
app_point = np.array([[d_app*np.cos(alpha)],
                      [d_app*np.sin(alpha)]]) + goal_pose
theta_app = robot.inverse_kinematic_geometry(app_point, elbow_option=0)

grid_size = 360
# calculate theta init index inside confuration 
theta_init_index = (map_multi_val(theta_init, -np.pi, np.pi, 0, grid_size)).astype(int)
# calculate theta goal index
theta_goal_index = (map_multi_val(theta_goal, -np.pi, np.pi, 0, grid_size)).astype(int)
# calculate theta approach index
theta_app_index = (map_multi_val(theta_app, -np.pi, np.pi, 0, grid_size)).astype(int)

# task space plot view
robot.plot_arm(theta_init, plt_basis=True)
robot.plot_arm(theta_goal)
robot.plot_arm(theta_app)
plt.plot(x_coord, y_coord)
obs_list = task_rectangle_obs_5()
for obs in obs_list:
    obs.plot()
plt.show()

# create config grid and view
map = configuration_generate_plannar_rr(robot, obs_list)
plt.imshow(map)
plt.show()

# Planning
x_init = node(theta_init_index[0,0], theta_init_index[1,0])
x_goal = node(theta_app_index[0,0], theta_app_index[1,0])
iteration = 1000
distance_weight = 0.5
obstacle_weight = 0.5
m = map.shape[0] * map.shape[1]
r = (2 * (1 + 1/2)**(1/2)) * (m/np.pi)**(1/2)
eta = r * (np.log(iteration) / iteration)**(1/2)
rrt = rrt_star(map, x_init, x_goal, eta, distance_weight, obstacle_weight, iteration)
np.random.seed(0)
rrt.start_planning()
path = rrt.Get_Path()

# Draw rrt tree
plt.imshow(map.T)
rrt.Draw_Tree()
rrt.Draw_path(path)
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
plt.plot(x_coord, y_coord)
robot.plot_arm(theta_goal)
for i in range(len(path)):
    # map index image to theta
    theta1 = map_val(pathx[i], 0, grid_size, -np.pi, np.pi)
    theta2 = map_val(pathy[i], 0, grid_size, -np.pi, np.pi)
    robot.plot_arm(np.array([[theta1], [theta2]]))
    plt.pause(1)
plt.show()