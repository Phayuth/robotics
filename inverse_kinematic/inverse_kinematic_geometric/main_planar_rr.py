import os
import sys

wd = os.path.abspath(os.getcwd())
sys.path.append(str(wd))

import matplotlib.pyplot as plt
import numpy as np
from robot.planar_rr import PlanarRR

# SECTION - create class robot
robot = PlanarRR()
# desiredPose = np.array([[0.0], [3.7]]) # aux1
# desiredPose = np.array([[0.15], [4-0.2590]]) # aux2
desiredPose = np.array([[-0.15], [4-0.2590]]) # aux3

# SECTION - find ik for both option
thetaUp = robot.inverse_kinematic_geometry(desiredPose, elbow_option=0)
print(f"==>> thetaUp: \n{thetaUp}")
thetaDown = robot.inverse_kinematic_geometry(desiredPose, elbow_option=1)
print(f"==>> thetaDown: \n{thetaDown}")

# SECTION - plot task space
robot.plot_arm(thetaUp)
plt.show()