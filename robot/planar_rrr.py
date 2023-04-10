"""Robot Class for Planar RRR

Reference:
- Inverse Kinematic : 
    1. https://www.youtube.com/watch?v=NjAAKruKiQM
    2. https://github.com/AymenHakim99/Forward-and-Inverse-Kinematics-for-3-DOF-Robotic-arm

"""

import numpy as np
import matplotlib.pyplot as plt


class PlanarRRR:

    def __init__(self):
        self.a1 = 1
        self.a2 = 1
        self.a3 = 0.5

    def forward_kinematic(self, theta, return_link_pos=False):
        theta1 = theta[0, 0]
        theta2 = theta[1, 0]
        theta3 = theta[2, 0]

        H01 = np.array([[np.cos(theta1), -np.sin(theta1), 0, 0], [np.sin(theta1), np.cos(theta1), 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

        H12 = np.array([[np.cos(theta2), -np.sin(theta2), 0, self.a1], [np.sin(theta2), np.cos(theta2), 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

        H23 = np.array([[np.cos(theta3), -np.sin(theta3), 0, self.a2], [np.sin(theta3), np.cos(theta3), 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

        H34 = np.array([[1, 0, 0, self.a3], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

        H04 = H01 @ H12 @ H23 @ H34
        phi = theta1 + theta2 + theta3

        if return_link_pos:

            # option for return link end pose. normally used for collision checking
            link_end_pose = []
            link_end_pose.append([0, 0])

            # link 1 pose
            x1 = self.a1 * np.cos(theta1)
            y1 = self.a1 * np.sin(theta1)
            link_end_pose.append([x1, y1])

            # link 2 pose
            x2 = self.a1 * np.cos(theta1) + self.a2 * np.cos(theta1 + theta2)
            y2 = self.a1 * np.sin(theta1) + self.a2 * np.sin(theta1 + theta2)
            link_end_pose.append([x2, y2])

            # link 3 pose
            x3 = self.a1 * np.cos(theta1) + self.a2 * np.cos(theta1 + theta2) + self.a3 * np.cos(theta1 + theta2 + theta3)
            y3 = self.a1 * np.sin(theta1) + self.a2 * np.sin(theta1 + theta2) + self.a3 * np.sin(theta1 + theta2 + theta3)
            link_end_pose.append([x3, y3])

            return link_end_pose

        else:
            return np.array([[H04[0, 3], H04[1, 3], phi]])

    def jacobian(self, theta):
        theta1 = theta[0, 0]
        theta2 = theta[1, 0]
        theta3 = theta[2, 0]

        J = np.array([[-self.a1 * np.sin(theta1) - self.a2 * np.sin(theta1 + theta2) - self.a3 * np.sin(theta1 + theta2 + theta3), -self.a2 * np.sin(theta1 + theta2) - self.a3 * np.sin(theta1 + theta2 + theta3),-self.a3 * np.sin(theta1 + theta2 + theta3)],
                      [ self.a1 * np.cos(theta3) + self.a2 * np.cos(theta1 + theta2) + self.a3 * np.cos(theta1 + theta2 + theta3),  self.a2 * np.cos(theta1 + theta2) + self.a3 * np.cos(theta1 + theta2 + theta3), self.a3 * np.cos(theta1 + theta2 + theta3)],
                      [                                                                                                         1,                                                                               1,                                         1]])

        return J

    def plot_arm(self, theta, plt_basis=False, plt_show=False):

        theta1 = theta[0, 0]
        theta2 = theta[1, 0]
        theta3 = theta[2, 0]

        H01 = np.array([[np.cos(theta1), -np.sin(theta1), 0, 0], [np.sin(theta1), np.cos(theta1), 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

        H12 = np.array([[np.cos(theta2), -np.sin(theta2), 0, self.a1], [np.sin(theta2), np.cos(theta2), 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

        H23 = np.array([[np.cos(theta3), -np.sin(theta3), 0, self.a2], [np.sin(theta3), np.cos(theta3), 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

        H34 = np.array([[1, 0, 0, self.a3], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

        p1 = H01
        p2 = H01 @ H12
        p3 = H01 @ H12 @ H23
        p4 = H01 @ H12 @ H23 @ H34

        if plt_basis:
            plt.axes().set_aspect('equal')
            plt.axvline(x=0, c="black")
            plt.axhline(y=0, c="black")

        plt.plot([p1[0, 3], p2[0, 3]], [p1[1, 3], p2[1, 3]], c="blue", linewidth=3)  # link 1
        plt.plot([p2[0, 3], p3[0, 3]], [p2[1, 3], p3[1, 3]], c="red", linewidth=3)  # link 2
        plt.plot([p3[0, 3], p4[0, 3]], [p3[1, 3], p4[1, 3]], c="brown", linewidth=3)  # link 3

        plt.scatter([p1[0, 3], p2[0, 3], p3[0, 3], p4[0, 3]], [p1[1, 3], p2[1, 3], p3[1, 3], p4[1, 3]], color='red')

        if plt_show:
            plt.show()

    def inverse_kinematic_geometry(self, desired_config, elbow_option=0):

        x = desired_config[0, 0]
        y = desired_config[1, 0]
        phi = desired_config[2, 0]

        x2 = x - self.a3 * np.cos(phi)
        y2 = y - self.a3 * np.sin(phi)

        t2term = (x2**2 + y2**2 - self.a1**2 - self.a2**2) / (2 * self.a1 * self.a2)

        if elbow_option == 0:
            sign = -1
        elif elbow_option == 1:
            sign = 1

        theta2 = sign * np.arccos(t2term)  # positive for elbow down, negative for elbow up

        t1term = ((self.a1 + self.a2 * np.cos(theta2)) * x2 + self.a2 * np.sin(theta2) * y2) / (x2**2 + y2**2)
        theta1 = np.arccos(t1term)

        theta3 = phi - theta1 - theta2

        return np.array([[theta1], [theta2], [theta3]])
