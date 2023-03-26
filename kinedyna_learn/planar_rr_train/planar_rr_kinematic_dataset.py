import os
import sys
wd = os.path.abspath(os.getcwd())
sys.path.append(str(wd))

import numpy as np
from robot.planar_rr import planar_rr

def planar_rr_generate_dataset(robot):

    # start sample
    sample_size = 360
    theta_candidate = np.linspace(-np.pi, np.pi, sample_size)

    sample_theta = []            # X sample_theta
    sample_endeffector_pose = [] # y sample_endeffector_pose

    for i in range(sample_size):
        for j in range(sample_size):
            endeffector_pose = robot.forward_kinematic(np.array([[theta_candidate[i]], [theta_candidate[j]]])) # this is where a machine learning for forward dynamic will replace

            sample_theta_row = [theta_candidate[i], theta_candidate[j]]
            sample_theta.append(sample_theta_row)

            sample_endeffector_row = [endeffector_pose[0,0], endeffector_pose[1,0]]
            sample_endeffector_pose.append(sample_endeffector_row)

    return np.array(sample_theta), np.array(sample_endeffector_pose)

if __name__=="__main__":

    robot = planar_rr()

    X, y = planar_rr_generate_dataset(robot)
    
    print("==>> sample_theta.shape: \n", X.shape)
    print("==>> sample_endeffector_pose.shape: \n", y.shape)