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
r = UR5e()
jorgn = np.array([0.5,0.5,-0.5,0,0,0]).reshape(6,1)
Torgn = r.forward_kinematic(jorgn, return_full_H=True)  # T06
print("==>> Torgn: \n", Torgn)


# SECTION - inverse kinematic
theta_ik = r.inverse_kinematic_geometry(Torgn)
print("==>> theta_ik: \n", theta_ik.T)
# for i in range(8):
#     th = theta_ik[:,i].reshape(6,1)
#     print(f"==>> th {i+1}th: \n", th.T)
    # Tik = r.forward_kinematic(th, return_full_H=True)
    # print("==>> Tik: \n", Tik)
