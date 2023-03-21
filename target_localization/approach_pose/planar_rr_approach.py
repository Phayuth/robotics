import os
import sys
wd = os.path.abspath(os.getcwd())
sys.path.append(str(wd))

import numpy as np
import matplotlib.pyplot as plt
from robot.planar_rr import planar_rr
from config_space_2d.generate_config_planar_rr import configuration_generate_plannar_rr
from map.taskmap_geo_format import task_rectangle_obs_1
from map.map_value_range import map_val

main_pose = 1.5

def polar2cats(r,theta):
    x = r*np.cos(theta) + main_pose
    y = r*np.sin(theta) + main_pose
    return x,y

# Create canidate pose
theta = np.linspace(np.pi/2,3*np.pi/2,10)
radius = 0.5
x_coord, y_coord = polar2cats(radius,theta)
obs_list = task_rectangle_obs_1()

# create robot instance
robot = planar_rr()
theta_ik = [robot.inverse_kinematic_geometry(np.array([[x_coord[i]],[y_coord[i]]]), elbow_option=0) for i in range(len(x_coord))]
theta_ik_main = robot.inverse_kinematic_geometry(np.array([[main_pose],[main_pose]]), elbow_option=0)

# create config space
grid_np = configuration_generate_plannar_rr(robot, obs_list)

# setup plot look
plt.axes().set_aspect('equal')
plt.axvline(x=0, c="black")
plt.axhline(y=0, c="black")

# plot data
plt.plot(x_coord,y_coord,c="green")
for obs in obs_list:
    obs.plot()
for each_theta in range(len(theta_ik)):
    robot.plot_arm(np.array([[theta_ik[each_theta][0].item()],[theta_ik[each_theta][1].item()]]))
plt.show()

theta1_plt = []
theta2_plt = []
for k in range(len(theta_ik)):
    theta1_index = int(map_val(theta_ik[k][0].item(), -np.pi, np.pi, 0, 360)) 
    theta2_index = int(map_val(theta_ik[k][1].item(), -np.pi, np.pi, 0, 360))
    theta1_plt.append(theta1_index)
    theta2_plt.append(theta2_index)

theta1_main_plt = int(map_val(theta_ik_main[0].item(), -np.pi, np.pi, 0, 360))
theta2_main_plt = int(map_val(theta_ik_main[1].item(), -np.pi, np.pi, 0, 360))

for u in range(len(theta1_plt)):
    grid_np[theta1_plt[u], theta2_plt[u]] = 2

grid_np[theta1_main_plt, theta2_main_plt] = 3

plt.imshow(grid_np)
plt.show()

print(theta1_plt)
print(theta2_plt)