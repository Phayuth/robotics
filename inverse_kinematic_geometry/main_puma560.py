""" Inverse Kinematic based on geometry derived. Main script to apply code from robot class
- Robot Type : Puma 560
- DOF : 6
- Return : 8 possible vector of theta
"""

import os
import sys

wd = os.path.abspath(os.getcwd())
sys.path.append(str(wd))

import numpy as np
from robot.puma560 import Puma560
np.set_printoptions(suppress=True)

# SECTION - create class robot
robot = Puma560()

thetaOriginal = np.array([2, 1, 0.5, 2, 1, 0]).reshape(6, 1)
print("==>> thetaOriginal: \n", thetaOriginal.reshape(1,6))

TForwardOriginal = robot.forward_kinematic(thetaOriginal, return_full_H=True)
print("==>> TForwardOriginal: \n", TForwardOriginal)

thetaIk = robot.inverse_kinematic_geometry(TForwardOriginal)
# print("==>> thetaIk: \n", thetaIk)

for i in range(8):
    possibleTheta = thetaIk[:, i].reshape(6, 1)
    TAfterIk = robot.forward_kinematic(possibleTheta, return_full_H=True)
    print(f"==>> Solution Number {i+1}: {thetaIk[:,i]}",)
    print(TAfterIk)