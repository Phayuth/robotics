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
from rigid_body_transformation.rotation_matrix import rotx, roty, rotz
from robot.ur5e import UR5e

# SECTION - create class robot
r = UR5e()
joints = np.array([[0], [0], [0], [0], [0], [0]])
g_S_T = r.forward_kinematic(joints, return_full_H=True)  # T06

gg = np.array([[1, 0, 0, 0.2],
               [0, 1, 0, 0.2],
               [0, 0, 1, 0.2],
               [0, 0, 0,   1]])


# SECTION - find ik
theta_ik = r.inverse_kinematic_geometry(gg)
print("==>> theta_ik: ", theta_ik)

# SECTION - plot task space
theta_result = theta_ik[:, 0]
theta_result = theta_result.reshape(6, 1)
print("==>> theta_result: ", theta_result)
r.plot_arm(theta_result, plt_basis=True, plt_show=True)
