import os
import sys

wd = os.path.abspath(os.getcwd())
sys.path.append(str(wd))

import matplotlib.pyplot as plt
import numpy as np
from robot.planar_rrr import PlanarRRR

# SECTION - create class robot
robot = PlanarRRR()
desiredPose = np.array([[1], [1], [0.2]])

# SECTION - find ik for both option
thetaUp = robot.inverse_kinematic_geometry(desiredPose, elbow_option=0)
thetaDown = robot.inverse_kinematic_geometry(desiredPose, elbow_option=1)

# SECTION - plot task space
robot.plot_arm(thetaUp)
robot.plot_arm(thetaDown)
plt.show()