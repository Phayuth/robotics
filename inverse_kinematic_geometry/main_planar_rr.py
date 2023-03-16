import os
import sys
wd = os.path.abspath(os.getcwd())
sys.path.append(str(wd))

from robot.planar_rr import planar_rr
import matplotlib.pyplot as plt
import numpy as np

# create class robot
robot = planar_rr()
desired_pose = np.array([[1],[1]])
# find ik
theta_up = robot.inverse_kinematic_geometry(desired_pose, elbow_option=0)
theta_down = robot.inverse_kinematic_geometry(desired_pose, elbow_option=1)
# plot
robot.plot_arm(theta_up)
plt.show()