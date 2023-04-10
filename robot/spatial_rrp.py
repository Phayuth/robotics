import numpy as np


class SpatialRRP:

    def __init__(self):
        self.a1 = 2
        self.a2 = 2
        self.a3 = 2
        self.a4 = 2

    def forward_kinematic(self, actuation):
        theta1 = actuation[0, 0]
        theta2 = actuation[0, 1]
        d3 = actuation[0, 2]

        H_1_to_0 = np.array([[np.cos(theta1),                  0,    np.sin(theta1),    self.a2*np.cos(theta1)],
                             [np.sin(theta1),                  0,   -np.cos(theta1),    self.a2*np.sin(theta1)],
                             [             0,                  1,                 0,                   self.a1],
                             [             0,                  0,                 0,                        1]])

        H_2_to_1 = np.array([[             0,    -np.sin(theta2),    np.cos(theta2),    self.a3*np.cos(theta2)],
                             [             0,     np.cos(theta2),    np.sin(theta2),    self.a3*np.sin(theta2)],
                             [            -1,                  0,                 0,                         0],
                             [             0,                  0,                 0,                        1]])

        H_3_to_2 = np.array([[             1,                  0,                 0,                    0],
                             [             0,                  1,                 0,                    0],
                             [             0,                  0,                 1,           self.a4+d3],
                             [             0,                  0,                 0,                   1]])            

        H_3_to_0 = H_1_to_0 @ H_2_to_1 @ H_3_to_2

        p3 = np.array([[0], [0], [0], [1]])

        p0 = H_3_to_0 @ p3

        return p0
