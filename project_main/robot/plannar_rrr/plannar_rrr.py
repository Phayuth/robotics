import numpy as np
import matplotlib.pyplot as plt

class planar_rrr:
    def __init__(self):
        self.a1 = 2
        self.a2 = 2
        self.a3 = 2
        
    def forward_kinematic(self, theta):
        theta1 = theta[0,0]
        theta2 = theta[1,0]
        theta3 = theta[2,0]

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

        p1 = H01
        p2 = H01 @ H12
        p3 = H01 @ H12 @ H23
        p4 = H01 @ H12 @ H23 @ H34

        link = np.array([[p1[0,3],p1[1,3]],
                         [p2[0,3],p2[1,3]],
                         [p3[0,3],p3[1,3]],
                         [p4[0,3],p4[1,3]]])

        return link
    
    def plot_arm(self, theta):
        theta1 = theta[0,0]
        theta2 = theta[1,0]
        theta3 = theta[2,0]

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

        p1 = H01
        p2 = H01 @ H12
        p3 = H01 @ H12 @ H23
        p4 = H01 @ H12 @ H23 @ H34

        # setup plot look
        # plt.axes().set_aspect('equal')
        # plt.axvline(x=0, c="black")
        # plt.axhline(y=0, c="black")

        # plot data
        plt.plot([p1[0,3],p2[0,3]], [p1[1,3],p2[1,3]],c ="blue")  # link 1
        plt.plot([p2[0,3],p3[0,3]], [p2[1,3],p3[1,3]],c ="red")   # link 2
        plt.plot([p3[0,3],p4[0,3]], [p3[1,3],p4[1,3]],c ="brown") # link 3

        # plt.show()

    def inverse_kinematic_geo(self, desired_config):
        # https://www.youtube.com/watch?v=NjAAKruKiQM
        # https://github.com/AymenHakim99/Forward-and-Inverse-Kinematics-for-3-DOF-Robotic-arm
        x = desired_config[0,0]
        y = desired_config[1,0]
        phi = desired_config[2,0]

        x2 = x - self.a3*np.cos(phi)
        y2 = y - self.a3*np.sin(phi)

        t2term = (x2**2 + y2**2 - self.a1**2 - self.a2**2)/(2*self.a1*self.a2)
        theta2 = np.arccos(t2term)

        t1term = ((self.a1 + self.a2*np.cos(theta2))*x2 + self.a2*np.sin(theta2)*y2)/(x2**2 + y2**2)
        theta1 = np.arccos(t1term)

        theta3 = phi - theta1 - theta2

        return np.array([[theta1],[theta2],[theta3]])
