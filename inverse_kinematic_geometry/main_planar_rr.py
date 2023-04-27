""" Inverse Kinematic based on geometry derived. Main script to apply code from robot class
- Robot Type : Planar RR
- DOF : 2
- Option : Elbow up and Elbow Down
"""

import os
import sys

wd = os.path.abspath(os.getcwd())
sys.path.append(str(wd))

import matplotlib.pyplot as plt
import numpy as np
from robot.planar_rr import PlanarRR

# SECTION - create class robot
robot = PlanarRR()
desiredPose = np.array([[1], [1]])

# SECTION - find ik for both option
thetaUp = robot.inverse_kinematic_geometry(desiredPose, elbow_option=0)
thetaDown = robot.inverse_kinematic_geometry(desiredPose, elbow_option=1)

# SECTION - plot task space
robot.plot_arm(thetaUp)
plt.show()