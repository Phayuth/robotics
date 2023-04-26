import os
import sys

wd = os.path.abspath(os.getcwd())
sys.path.append(str(wd))

import matplotlib.pyplot as plt
import numpy as np
from util.coord_transform import polar2cats, circle_plt
from robot.planar_rrr import PlanarRRR

# create robot instance
robot = PlanarRRR()


# SECTION - user define pose and calculate theta
x_targ = 2.0
y_targ = 1.0
alpha_targ = 3.14  # given from grapse pose candidate
d_app = 0.5
phi_targ = alpha_targ - np.pi
target = np.array([x_targ, y_targ, phi_targ]).reshape(3, 1)
r_crop = 0.03

line_top_x_start = r_crop*np.cos(alpha_targ - np.pi/2) + x_targ
line_top_y_start = r_crop*np.sin(alpha_targ - np.pi/2) + y_targ
line_bot_x_start = r_crop*np.cos(alpha_targ + np.pi/2) + x_targ
line_bot_y_start = r_crop*np.sin(alpha_targ + np.pi/2) + y_targ

x_app = d_app * np.cos(target[2, 0] + np.pi) + target[0, 0]
y_app = d_app * np.sin(target[2, 0] + np.pi) + target[1, 0]

line_top_x_end = r_crop*np.cos(alpha_targ - np.pi/2) + x_app
line_top_y_end = r_crop*np.sin(alpha_targ - np.pi/2) + y_app
line_bot_x_end = r_crop*np.cos(alpha_targ + np.pi/2) + x_app
line_bot_y_end = r_crop*np.sin(alpha_targ + np.pi/2) + y_app


t_targ = robot.inverse_kinematic_geometry(target, elbow_option=0)


# SECTION - calculate approach pose and calculate theta
# aprch = np.array([[d_app * np.cos(target[2, 0] + np.pi) + target[0, 0]], [d_app * np.sin(target[2, 0] + np.pi) + target[1, 0]], [target[2, 0]]])
# t_app = robot.inverse_kinematic_geometry(aprch, elbow_option=0)


# SECTION - plot task space
robot.plot_arm(t_targ, plt_basis=True)
plt.plot([line_top_x_start, line_top_x_end], [line_top_y_start, line_top_y_end])
plt.plot([line_bot_x_start, line_bot_x_end], [line_bot_y_start, line_bot_y_end])
# circle_plt(x_targ, y_targ, r_crop)
# circle_plt(x_targ, y_targ, d_app)
plt.show()


from collision_check_geometry import collision_class

def configuration_generate_plannar_rrr(robot, obs_list):
    grid_size = 75
    theta1 = np.linspace(-np.pi, np.pi, grid_size)
    theta2 = np.linspace(-np.pi, np.pi, grid_size)
    theta3 = np.linspace(-np.pi, np.pi, grid_size)

    grid_map = np.zeros((grid_size, grid_size, grid_size))

    if obs_list == []:
        return 1 - grid_map
    
    else:
        for th1 in range(len(theta1)):
            for th2 in range(len(theta2)):
                for th3 in range(len(theta3)):
                    print(f"at theta1 {theta1[th1]} | at theta2 {theta2[th2]} | at theta3 {theta3[th3]}")
                    theta = np.array([[theta1[th1]], [theta2[th2]], [theta3[th3]]])
                    link_pose = robot.forward_kinematic(theta, return_link_pos=True)
                    linearm1 = collision_class.ObjLine2D(link_pose[0][0], link_pose[0][1], link_pose[1][0], link_pose[1][1])
                    linearm2 = collision_class.ObjLine2D(link_pose[1][0], link_pose[1][1], link_pose[2][0], link_pose[2][1])
                    linearm3 = collision_class.ObjLine2D(link_pose[2][0], link_pose[2][1], link_pose[3][0], link_pose[3][1])

                    col = []
                    for i in obs_list:
                        col1 = collision_class.intersect_line_v_line(linearm1, i)
                        col2 = collision_class.intersect_line_v_line(linearm2, i)
                        col3 = collision_class.intersect_line_v_line(linearm3, i)
                        col.extend((col1, col2, col3))

                    if True in col:
                        grid_map[th1, th2, th3] = 1

        return 1 - grid_map



linetop = collision_class.ObjLine2D(line_top_x_start, line_top_y_start, line_top_x_end, line_top_y_end)
linebot = collision_class.ObjLine2D(line_bot_x_start, line_bot_y_start, line_bot_x_end, line_bot_y_end)
obs_list = [linetop, linebot]

# map = configuration_generate_plannar_rrr(robot, obs_list)

# np.save('./map/mapdata/config_rrr_virtualobs.npy', map)
map = np.load('./map/mapdata/config_rrr_virtualobs.npy')

plt.imshow(map[:, :, 0])  # plt view of each index slice 3D into 2D image
plt.show()

from map.mapclass import map_val, map_vec
from planner.ready.rrt_2D.rrtstar_costmap_biassampling3d import node, rrt_star
from util.extract_path_class import extract_path_class_3d





# define task space init point and goal point
init_pose = np.array([[2.5],[0],[0]])

# using inverse kinematic, determine the theta configuration space in continuous space
theta_init = robot.inverse_kinematic_geometry(init_pose, elbow_option=0)
theta_goal = t_targ

# grid size
grid_size = 75
# calculate theta init index inside confuration 
x_init = map_vec(theta_init, -np.pi, np.pi, 0, grid_size)
x_goal = map_vec(theta_goal, -np.pi, np.pi, 0, grid_size)

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
plt.plot([line_top_x_start, line_top_x_end], [line_top_y_start, line_top_y_end])
plt.plot([line_bot_x_start, line_bot_x_end], [line_bot_y_start, line_bot_y_end])
circle_plt(x_targ, y_targ, r_crop)
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