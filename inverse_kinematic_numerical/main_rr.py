import os
import sys
wd = os.path.abspath(os.getcwd()) # get the top parent folder
sys.path.append(str(wd)) # add it to path

import numpy as np
import matplotlib.pyplot as plt

from numerical_jacinv import ik_jacobian_inverse
from numerical_jacpseudoinv import ik_jacobian_pseudo_inverse
from numerical_jactranspose import ik_jacobian_transpose
from numerical_ls_dls import ik_damped_leastsquare
from numerical_ls_sdls import ik_selectively_damped_leastsquare
from robot import planar_rr

# Create Robot Class
robot = planar_rr.planar_rr()

# Create Numerical IK Class
Dmax = 0.5
damp_cte = 0.5

ik_ji = ik_jacobian_inverse(100,robot)
ik_jp = ik_jacobian_pseudo_inverse(100,robot)
ik_jt = ik_jacobian_transpose(100,robot)
ik_dl = ik_damped_leastsquare(100,robot,damp_cte)
ik_sd = ik_selectively_damped_leastsquare(100,robot)

# Create Desired Pose and Current data
theta = np.array([[0.],[0.]]) # vector 2x1
x_desired = np.array([[0.5],[0.5]]) # vector 2x1

# Get Solution
# theta_solution = ik_ji.inverse_jac(theta,x_desired)
# theta_solution = ik_ji.inverse_jac_ClampMag(theta,x_desired,Dmax)
theta_solution = ik_jp.pseudoinverse_jac(theta,x_desired)
theta_solution = ik_jt.transpose_jac(theta,x_desired)

theta_solution = ik_dl.damped_least_square(theta,x_desired)
theta_solution = ik_dl.damped_least_square_ClampMag(theta,x_desired,Dmax)
theta_solution = ik_sd.selectively_damped_least_square_fast(theta,x_desired)
# theta_solution = ik_sd.selectively_damped_least_square(theta,x_desired,np.pi/4)

# Show Solution
print(theta_solution)
robot.plot_arm(theta_solution)