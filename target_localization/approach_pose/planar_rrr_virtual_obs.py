import os
import sys

wd = os.path.abspath(os.getcwd())
sys.path.append(str(wd))

import matplotlib.pyplot as plt
import numpy as np

from collision_check_geometry import collision_class
from config_space_2d.generate_config_planar_rrr import configuration_generate_plannar_rrr
from robot.planar_rrr import PlanarRRR
from planner_util.coord_transform import circle_plt, polar2cats
from map.mapclass import map_val, map_vec
from planner.ready.rrt_2D.rrtstar_costmap_biassampling3d import rrt_star
from planner_util.extract_path_class import extract_path_class_3d

# create robot instance
robot = PlanarRRR()

# SECTION - user define pose and calculate theta
xTarg = 1.5
yTarg = 0.2
alphaTarg = 2  # given from grapse pose candidate
hD = 0.25
wD = 0.25
rCrop = 0.1
phiTarg = alphaTarg - np.pi
target = np.array([xTarg, yTarg, phiTarg]).reshape(3, 1)

xTopStart = (rCrop + hD) * np.cos(alphaTarg - np.pi / 2) + xTarg
yTopStart = (rCrop + hD) * np.sin(alphaTarg - np.pi / 2) + yTarg
xBotStart = (rCrop) * np.cos(alphaTarg + np.pi / 2) + xTarg
yBotStart = (rCrop) * np.sin(alphaTarg + np.pi / 2) + yTarg
recTop = collision_class.ObjRec(xTopStart, yTopStart, hD, wD, angle=alphaTarg)
recBot = collision_class.ObjRec(xBotStart, yBotStart, hD, wD, angle=alphaTarg)

thetaGoal = robot.inverse_kinematic_geometry(target, elbow_option=0)

# SECTION - plot task space
robot.plot_arm(thetaGoal, plt_basis=True)
recTop.plot()
recBot.plot()
circle_plt(xTarg, yTarg, radius=rCrop)
plt.show()

obsList = [recTop, recBot]
# map = configuration_generate_plannar_rrr(robot, obsList)
# np.save('./map/mapdata/config_rrr_virtualobs.npy', map)
# map = np.load('./map/mapdata/config_rrr_virtualobs.npy')

# define task space init point and goal point
initPose = np.array([[2.5], [0], [0]])

# using inverse kinematic, determine the theta configuration space in continuous space
thetaInit = robot.inverse_kinematic_geometry(initPose, elbow_option=0)

gridSize = 75
xInit = map_vec(thetaInit, -np.pi, np.pi, 0, gridSize)
xGoal = map_vec(thetaGoal, -np.pi, np.pi, 0, gridSize)

rrt = rrt_star(map, xInit, xGoal, w1=0.5, w2=0.5, maxiteration=1000)
np.random.seed(0)
rrt.start_planning()
path = rrt.Get_Path()
rrt.Draw_tree()
rrt.Draw_path(path)
plt.show()

pathx, pathy, pathz = extract_path_class_3d(path)
print("==>> pathx: ", pathx)
print("==>> pathy: ", pathy)
print("==>> pathz: ", pathz)
recTop.plot()
recBot.plot()
for i in range(len(path)):
    # map index image to theta
    theta1 = map_val(pathx[i], 0, gridSize, -np.pi, np.pi)
    theta2 = map_val(pathy[i], 0, gridSize, -np.pi, np.pi)
    theta3 = map_val(pathz[i], 0, gridSize, -np.pi, np.pi)
    robot.plot_arm(np.array([[theta1], [theta2], [theta3]]))
    plt.pause(1)
plt.show()