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
from rigid_body_transformation.rotation_matrix import rotx, roty, rotz
from robot.puma560 import Puma560
np.set_printoptions(suppress=True)

# SECTION - create class robot
robot = Puma560()

theta_original = np.array([2, 1, 0.5, 2, 1, 0]).reshape(6, 1)
print("==>> theta_original: \n", theta_original)

Tforward_original = robot.forward_kinematic(theta_original, return_full_H=True)
print("==>> Tforward_original: \n", Tforward_original)

theta_ik = robot.inverse_kinematic_geometry(Tforward_original)

for i in range(8):
    tt = theta_ik[:, i].reshape(6, 1)
    Tforward_after_ik = robot.forward_kinematic(tt, return_full_H=True)
    print(f"==>> Solution Number {i+1}: {theta_ik[:,i]}",)
    print(Tforward_after_ik)