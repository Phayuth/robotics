import numpy as np

class planar_rrr:
    def __init__(self):
        self.a1 = 2
        self.a2 = 2
        self.a3 = 2
        
    def forward_kinematic(self,theta):
        theta1 = theta[0,0]
        theta2 = theta[0,1]
        theta3 = theta[0,2]

        H01 = np.array([[np.cos(theta1),    -np.sin(theta1),    0,    0],
                        [np.sin(theta1),     np.cos(theta1),    0,    0],
                        [             0,                  0,    1,    0],
                        [             0,                  0,    0,    1]])

        H12 = np.array([[np.cos(theta2),    -np.sin(theta2),    0,    self.a1],
                        [np.sin(theta2),     np.cos(theta2),    0,          0],
                        [             0,                  0,    1,          0],
                        [             0,                  0,    0,          1]])

        H23 = np.array([[np.cos(theta3),    -np.sin(theta3),    0,    self.a2],
                        [np.sin(theta3),     np.cos(theta3),    0,          0],
                        [             0,                  0,    1,          0],
                        [             0,                  0,    0,          1]])
            
        H34 = np.array([[             1,                  0,    0,    self.a3],
                        [             0,                  1,    0,          0],
                        [             0,                  0,    1,          0],
                        [             0,                  0,    0,          1]])

        H04 = H01 @ H12 @ H23 @ H34

        p4 = np.array([[0],[0],[0],[1]])

        p0 = H04 @ p4

        return p0
