import os
import sys
wd = os.path.abspath(os.getcwd())
sys.path.append(str(wd))

from robot.ur5e import ur5e
import numpy as np
from rigid_body_transformation.rotation_matrix import rotx, roty, rotz

# Test
r = ur5e()
joints = np.array([[0],[0],[0],[0],[0],[0]])
g_S_T = r.forward_kinematic(joints, return_full_H=True) # T06

gg = np.array([[1, 0, 0, 0.2],
               [0, 1, 0, 0.2],
               [0, 0, 1, 0.2],
               [0, 0, 0,   1]])

theta_ik = r.inverse_kinematic_geometry(gg)
print("==>> theta_ik: ", theta_ik)

theta_result = theta_ik[:,0]
theta_result = theta_result.reshape(6,1)
print("==>> theta_result: ", theta_result)
r.plot_arm(theta_result, plt_basis=True, plt_show=True)
