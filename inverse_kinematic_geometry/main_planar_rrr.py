import os
import sys
wd = os.path.abspath(os.getcwd())
sys.path.append(str(wd))

from robot.planar_rrr import planar_rrr
import matplotlib.pyplot as plt
import numpy as np

# create class robot
robot = planar_rrr()
desired_pose = np.array([[1],[1],[0.2]])
# find ik
theta_up = robot.inverse_kinematic_geometry(desired_pose, elbow_option=0)
theta_down = robot.inverse_kinematic_geometry(desired_pose, elbow_option=1)
# plot
robot.plot_arm(theta_up)
robot.plot_arm(theta_down)
plt.show()