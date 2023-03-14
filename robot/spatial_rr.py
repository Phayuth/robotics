import numpy as np

class spatial_rr:
    def __init__(self):
        self.alpha1 = np.pi/2
        self.alpha2 = 0
        self.d1 = 1
        self.d2 = 0
        self.a1 = 0
        self.a2 = 1

    def dh_transformation(theta,alpha,d,a):
        R = np.array([[np.cos(theta), -np.sin(theta)*np.cos(alpha),  np.sin(theta)*np.sin(alpha), a*np.cos(theta)],
                      [np.sin(theta),  np.cos(theta)*np.cos(alpha), -np.cos(theta)*np.sin(alpha), a*np.sin(theta)],
                      [            0,                np.sin(alpha),                np.cos(alpha),               d],
                      [            0,                            0,                            0,               1]])
        return R

    def forward_kinematic(self,theta):
        theta1 = theta[0,0]
        theta2 = theta[1,0]

        A1 = self.dh_transformation(theta1,self.alpha1,self.d1,self.a1)
        A2 = self.dh_transformation(theta2,self.alpha2,self.d2,self.a2)

        T = A1 @ A2
        p2 = np.array([[0],[0],[0],[1]]) # the fourth element MUST be equal to 1
        p0 = T @ p2

        return p0
