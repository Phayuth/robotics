import os
import sys
wd = os.path.abspath(os.getcwd())
sys.path.append(str(wd))

import numpy as np
from robot.planar_rr import PlanarRR
from planar_rr_kinematic_dataset import planar_rr_generate_dataset

# create robot class
robot = PlanarRR()

# create dataset of forward kinematic
sample_theta, sample_endeffector_pose = planar_rr_generate_dataset(robot)

# define desired value of end point
desired_value = np.array([[1],[1]])

# create an error vector
error_vector = []
for k in range(len(sample_endeffector_pose)):
    end_pose = np.array([[sample_endeffector_pose[k,0]],[sample_endeffector_pose[k,1]]])
    error = end_pose - desired_value # find error between all possible endpoint and desired value
    norm = np.linalg.norm(error)     # find norm of error
    error_vector.append([norm])

error_vector = np.array(error_vector)
print("==>> error_vector: \n", error_vector)

policy = np.argmin(error_vector) # policy find the minimum error and return its index
print("==>> policy: \n", policy)

theta_result = sample_theta[policy].reshape(2,1) # use the value from policy to find the correspond sample with the same index. and get it action to input to the real system
print("==>> theta_result: \n", theta_result)

robot.plot_arm(theta_result, plt_basis=True, plt_show=True) # input to the system