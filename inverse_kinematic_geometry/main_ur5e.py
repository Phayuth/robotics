""" Inverse Kinematic based on geometry derived. Main script to apply code from robot class
- Robot Type : UR5es
- DOF : 6
- Return : 8 possible vector of theta
"""

import os
import sys

wd = os.path.abspath(os.getcwd())
sys.path.append(str(wd))

import numpy as np
from robot.ur5e import UR5e

# SECTION - create class robot
robot = UR5e()
jointOriginal = np.array([0.5, 0.5, -0.5, 0, 0, 0]).reshape(6, 1)
TOriginal = robot.forward_kinematic(jointOriginal, return_full_H=True)  # T06
print("==>> TOriginal: \n", TOriginal)

# SECTION - inverse kinematic
thetaIk = robot.inverse_kinematic_geometry(TOriginal)
print("==>> thetaIk: \n", thetaIk.T)
# for i in range(8):
#     possibleTheta = thetaIk[:,i].reshape(6,1)
#     print(f"==>> possibleTheta {i+1}possibleTheta: \n", possibleTheta.T)
#     TAfterIk = robot.forward_kinematic(possibleTheta, return_full_H=True)
#     print("==>> TAfterIk: \n", TAfterIk)
