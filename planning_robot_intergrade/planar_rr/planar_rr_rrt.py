import os
import sys

wd = os.path.abspath(os.getcwd())
sys.path.append(str(wd))

import matplotlib.pyplot as plt
import numpy as np

from config_space_2d.generate_config_planar_rr import configuration_generate_plannar_rr
from map.mapclass import CostMapClass, CostMapLoader, GeoMapClass
from map.taskmap_geo_format import task_rectangle_obs_1
from planner.rrtbase import RRTBase
from robot.planar_rr import PlanarRR
from util.extract_path_class import extract_path_class_2d

robot = PlanarRR()

start_pose = np.array([1.8, 3.3]).reshape(2, 1)
goal_pose = np.array([3.5, 1.8]).reshape(2, 1)

start_theta = robot.inverse_kinematic_geometry(start_pose, elbow_option=0)
goal_theta = robot.inverse_kinematic_geometry(goal_pose, elbow_option=0)

robot.plot_arm(start_theta, plt_basis=True)
robot.plot_arm(goal_theta)
obs_list = task_rectangle_obs_1()
for obs in obs_list:
    obs.plot()
plt.show()

maparray = configuration_generate_plannar_rr(robot, obs_list)
maploader = CostMapLoader.loadarray(maparray)
mapclass = CostMapClass(maploader=maploader, maprange=[[-np.pi, np.pi], [-np.pi, np.pi]])

planner = RRTBase(mapclass, start_theta, goal_theta, eta=0.1, maxiteration=2000)
planner.plot_env()
plt.show()
planner.planing()
planner.plot_env(after_plan=True)
path = planner.search_path()
plt.plot([node.x for node in path], [node.y for node in path], color='blue')
plt.show()

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
    robot.plot_arm(np.array([[pathx[i]], [pathy[i]]]))
    plt.pause(1)
plt.show()