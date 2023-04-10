""" Inverse Kinematic based on geometry derived. Main script to apply code from robot class
- Robot Type : Planar RRR
- DOF : 3
- Option : Elbow up and Elbow Down
"""

import os
import sys

wd = os.path.abspath(os.getcwd())
sys.path.append(str(wd))

import matplotlib.pyplot as plt
import numpy as np
from robot.planar_rrr import PlanarRRR

# SECTION - create class robot
robot = PlanarRRR()
desired_pose = np.array([[1], [1], [0.2]])

# SECTION - find ik for both option
theta_up = robot.inverse_kinematic_geometry(desired_pose, elbow_option=0)
theta_down = robot.inverse_kinematic_geometry(desired_pose, elbow_option=1)

# SECTION - plot task space
robot.plot_arm(theta_up)
robot.plot_arm(theta_down)
plt.show()