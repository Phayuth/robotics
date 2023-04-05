import os
import sys
wd = os.path.abspath(os.getcwd())
sys.path.append(str(wd))

import numpy as np
import matplotlib.pyplot as plt
from rigid_body_transformation.homogeneous_transformation import inverse_hom_trans as invht
ax = plt.axes(projection='3d')


class puma560:
    def __init__(self) -> None:
        # DH parameter is from introduction to robotics mechanic and control by J.J. Craig for modified dh version
        self.a     = np.array([      [0], [-0.425],  [-0.392],       [0],        [0],      [0]])
        self.alpha = np.array([      [0], [-np.pi],       [0],  [-np.pi],    [np.pi], [-np.pi]])
        self.d     = np.array([  [0.089],      [0],       [0],   [0.109],    [0.094],  [0.082]])

        # DH parameter is from introduction to robotics mechanic and control by J.J. Craig for original dh version
        self.a     = np.array([      [0], [-0.425],  [-0.392],       [0],        [0],      [0]])
        self.alpha = np.array([      [0], [-np.pi],       [0],  [-np.pi],    [np.pi], [-np.pi]])
        self.d     = np.array([  [0.089],      [0],       [0],   [0.109],    [0.094],  [0.082]])

    def dh_transformation_mod(theta, alpha, d, a): # modified dh method from craig
        R = np.array([[              np.cos(theta),              -np.sin(theta),              0,                a],
                      [np.sin(theta)*np.cos(alpha), np.cos(theta)*np.cos(alpha), -np.sin(alpha), -np.sin(alpha)*d],
                      [np.sin(theta)*np.sin(alpha), np.cos(theta)*np.sin(alpha),  np.cos(alpha),  np.cos(alpha)*d],
                      [                          0,                           0,              0,                1]])
        return R
    
    def forward_kinematic(self, theta, return_full_H=False, return_each_H=False):

        A1 = self.dh_transformation_mod(theta[0,0],self.alpha[0,0],self.d[0,0],self.a[0,0])
        A2 = self.dh_transformation_mod(theta[1,0],self.alpha[1,0],self.d[1,0],self.a[1,0])
        A3 = self.dh_transformation_mod(theta[2,0],self.alpha[2,0],self.d[2,0],self.a[2,0])
        A4 = self.dh_transformation_mod(theta[3,0],self.alpha[3,0],self.d[3,0],self.a[3,0])
        A5 = self.dh_transformation_mod(theta[4,0],self.alpha[4,0],self.d[4,0],self.a[4,0])
        A6 = self.dh_transformation_mod(theta[5,0],self.alpha[5,0],self.d[5,0],self.a[5,0])

        T06 = A1 @ A2 @ A3 @ A4 @ A5 @ A6
        
        if return_full_H: # option to return transformation from base to end effector
            return T06
        if return_each_H: # option to return all transformation
            return A1, A2, A3, A4, A5, A6
