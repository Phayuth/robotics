""" Robot Class for UR5 and UR5e

Reference:
- DH Parameter
    1. DH parameter, Forward and Inverse Kinematic is from : http://rasmusan.dk/wp-content/uploads/ur5_kinematics.pdf

- Convert TF to RPY
    # x - y - z sequence
    # tan(roll) = r32/r33
    # tan(pitch)= -r31/(sqrt(r32^2 + r33^2))
    # tan(yaw)  = r21/r11
    # np.arctan2(y, x)

"""

import os
import sys

wd = os.path.abspath(os.getcwd())
sys.path.append(str(wd))

import numpy as np
from rigid_body_transformation.rigid_trans import RigidBodyTransformation as rbt


class UR5e:

    def __init__(self):
        # DH
        self.alpha = np.array([     [0], [np.pi/2],      [0],      [0], [np.pi/2], [-np.pi/2]])
        self.a     = np.array([     [0],       [0], [-0.425], [-0.392],       [0],        [0]])
        self.d     = np.array([ [0.089],       [0],      [0],  [0.109],   [0.094],    [0.082]])

    def forward_kinematic(self, theta, return_full_H=False, return_each_H=False):
        T01 = rbt.dh_transformation_mod(theta[0, 0], self.alpha[0, 0], self.d[0, 0], self.a[0, 0])
        T12 = rbt.dh_transformation_mod(theta[1, 0], self.alpha[1, 0], self.d[1, 0], self.a[1, 0])
        T23 = rbt.dh_transformation_mod(theta[2, 0], self.alpha[2, 0], self.d[2, 0], self.a[2, 0])
        T34 = rbt.dh_transformation_mod(theta[3, 0], self.alpha[3, 0], self.d[3, 0], self.a[3, 0])
        T45 = rbt.dh_transformation_mod(theta[4, 0], self.alpha[4, 0], self.d[4, 0], self.a[4, 0])
        T56 = rbt.dh_transformation_mod(theta[5, 0], self.alpha[5, 0], self.d[5, 0], self.a[5, 0])

        T06 = T01 @ T12 @ T23 @ T34 @ T45 @ T56

        if return_full_H:
            # option to return transformation from base to end effector
            return T06

        if return_each_H:
            # option to return all transformation
            return T01, T12, T23, T34, T45, T56

        if not return_each_H and not return_each_H:
            # option to return 6 X 1 vector pose
            roll = np.arctan2(T06[2, 1], T06[2, 2])
            pitch = np.arctan2(-T06[2, 0], np.sqrt(T06[2, 1] * T06[2, 1] + T06[2, 2] * T06[2, 2]))
            yaw = np.arctan2(T06[1, 0], T06[0, 0])
            x_current = np.array([[T06[0, 3]], [T06[1, 3]], [T06[2, 3]], [roll], [pitch], [yaw]])

            return x_current

    def jacobian(self, theta):
        A1 = rbt.dh_transformation_mod(theta[0, 0], self.alpha[0, 0], self.d[0, 0], self.a[0, 0])
        A2 = rbt.dh_transformation_mod(theta[1, 0], self.alpha[1, 0], self.d[1, 0], self.a[1, 0])
        A3 = rbt.dh_transformation_mod(theta[2, 0], self.alpha[2, 0], self.d[2, 0], self.a[2, 0])
        A4 = rbt.dh_transformation_mod(theta[3, 0], self.alpha[3, 0], self.d[3, 0], self.a[3, 0])
        A5 = rbt.dh_transformation_mod(theta[4, 0], self.alpha[4, 0], self.d[4, 0], self.a[4, 0])
        A6 = rbt.dh_transformation_mod(theta[5, 0], self.alpha[5, 0], self.d[5, 0], self.a[5, 0])

        T01 = A1
        T02 = A1 @ A2
        T03 = A1 @ A2 @ A3
        T04 = A1 @ A2 @ A3 @ A4
        T05 = A1 @ A2 @ A3 @ A4 @ A5
        T06 = A1 @ A2 @ A3 @ A4 @ A5 @ A6

        Z0 = np.array([[0], [0], [1]])
        Z1 = np.array([[T01[0, 2]], [T01[1, 2]], [T01[2, 2]]])
        Z2 = np.array([[T02[0, 2]], [T02[1, 2]], [T02[2, 2]]])
        Z3 = np.array([[T03[0, 2]], [T03[1, 2]], [T03[2, 2]]])
        Z4 = np.array([[T04[0, 2]], [T04[1, 2]], [T04[2, 2]]])
        Z5 = np.array([[T05[0, 2]], [T05[1, 2]], [T05[2, 2]]])

        O0 = np.array([[0], [0], [0]])
        O1 = np.array([[T01[3, 0]], [T01[3, 1]], [T01[3, 2]]])
        O2 = np.array([[T02[3, 0]], [T02[3, 1]], [T02[3, 2]]])
        O3 = np.array([[T03[3, 0]], [T03[3, 1]], [T03[3, 2]]])
        O4 = np.array([[T04[3, 0]], [T04[3, 1]], [T04[3, 2]]])
        O5 = np.array([[T05[3, 0]], [T05[3, 1]], [T05[3, 2]]])
        O6 = np.array([[T06[3, 0]], [T06[3, 1]], [T06[3, 2]]])

        Jv1 = np.transpose(np.cross(np.transpose(Z0), np.transpose(O6 - O0)))
        Jv2 = np.transpose(np.cross(np.transpose(Z1), np.transpose(O6 - O1)))
        Jv3 = np.transpose(np.cross(np.transpose(Z2), np.transpose(O6 - O2)))
        Jv4 = np.transpose(np.cross(np.transpose(Z3), np.transpose(O6 - O3)))
        Jv5 = np.transpose(np.cross(np.transpose(Z4), np.transpose(O6 - O4)))
        Jv6 = np.transpose(np.cross(np.transpose(Z5), np.transpose(O6 - O5)))

        Jw1 = Z0
        Jw2 = Z1
        Jw3 = Z2
        Jw4 = Z3
        Jw5 = Z4
        Jw6 = Z5

        J1 = np.vstack((Jv1, Jw1))
        J2 = np.vstack((Jv2, Jw2))
        J3 = np.vstack((Jv3, Jw3))
        J4 = np.vstack((Jv4, Jw4))
        J5 = np.vstack((Jv5, Jw5))
        J6 = np.vstack((Jv6, Jw6))

        J = np.hstack((J1, J2, J3, J4, J5, J6))

        return J

    def jacobian_analytical(self, theta, roll, pitch, yaw):
        B = np.array([[np.cos(yaw)*np.sin(pitch), -np.sin(yaw), 0],
                      [np.sin(yaw)*np.sin(pitch),  np.cos(yaw), 0],
                      [            np.cos(pitch),            0, 1]])

        Binv = np.linalg.inv(B)

        Ja_mul = np.array([[1, 0, 0,         0,         0,         0],
                           [0, 1, 0,         0,         0,         0],
                           [0, 0, 1,         0,         0,         0],
                           [0, 0, 0, Binv[0,0], Binv[0,1], Binv[0,2]],
                           [0, 0, 0, Binv[1,0], Binv[1,1], Binv[1,2]],
                           [0, 0, 0, Binv[2,0], Binv[2,1], Binv[2,2]]])

        Ja = Ja_mul @ self.jacobian(theta)

        return Ja

    def inverse_kinematic_geometry(self, goal_desired):
        T06 = goal_desired  # given 4x4 transformation

        theta = np.zeros((6, 8))

        # theta 1
        P05 = T06 @ np.array([0, 0, -self.d[5, 0], 1]).reshape(4, 1)
        p05x = P05[0, 0]
        p05y = P05[1, 0]
        p05xy = np.sqrt((p05x**2) + (p05y**2))
        if self.d[3, 0] > p05xy:
            print("no solution for th1")
        psi = np.arctan2(p05y, p05x)
        phi = np.arccos(self.d[3, 0] / p05xy)
        theta1_1 = psi + phi + np.pi / 2
        theta1_2 = psi - phi + np.pi / 2
        theta[0, 0], theta[0, 1], theta[0, 2], theta[0, 3] = theta1_1, theta1_1, theta1_1, theta1_1
        theta[0, 4], theta[0, 5], theta[0, 6], theta[0, 7] = theta1_2, theta1_2, theta1_2, theta1_2

        # theta5
        col = [0, 1, 4, 5]
        for i in range(8):
            p06x = T06[0, 3]
            p06y = T06[1, 3]
            theta5 = np.arccos((p06x * np.sin(theta[0, i]) - p06y * np.cos(theta[0, i]) - self.d[3, 0]) / self.d[5, 0])

            if abs(p06x * np.sin(theta[0, i]) - p06y * np.cos(theta[0, i]) - self.d[3, 0]) >= abs(self.d[5, 0]):
                print("no solution for th5")

            if i in col:
                theta[4, i] = theta5
            else:
                theta[4, i] = -theta5

        # theta6
        for i in range(8):
            T60 = rbt.h_inverse(T06)
            Xy60 = T60[1, 0]
            Yy60 = T60[1, 1]
            Xx60 = T60[0, 0]
            Yx60 = T60[0, 1]

            theta6_term1 = ((-Xy60 * np.sin(theta[0, i])) + (Yy60 * np.cos(theta[0, i]))) / np.sin(theta[4, i])
            theta6_term2 = ((Xx60 * np.sin(theta[0, i])) - (Yx60 * np.cos(theta[0, i]))) / np.sin(theta[4, i])
            theta6 = np.arctan2(theta6_term1, theta6_term2)

            if np.sin(theta[4, i]) == 0:
                print("theta6 solution is underdetermine in this case theta 6 can be random")
                theta6 = 0

            theta[5, i] = theta6

        # theta3
        col1 = [0, 2, 4, 6]
        col2 = [1, 3, 5, 7]
        for i in range(8):
            T01 = rbt.dh_transformation_mod(theta[0, i], self.alpha[0, 0], self.d[0, 0], self.a[0, 0])
            T45 = rbt.dh_transformation_mod(theta[4, i], self.alpha[4, 0], self.d[4, 0], self.a[4, 0])
            T56 = rbt.dh_transformation_mod(theta[5, i], self.alpha[5, 0], self.d[5, 0], self.a[5, 0])

            T46 = T45 @ T56
            T60 = rbt.h_inverse(T06)
            T40 = T46 @ T60
            T41 = T40 @ T01
            T14 = rbt.h_inverse(T41)

            p14x = T14[0, 3]
            p14z = T14[2, 3]

            p14xz = np.sqrt(p14x**2 + p14z**2)

            theta3 = np.arccos((p14xz**2 - (self.a[2, 0]**2) - (self.a[3, 0]**2)) / (2 * self.a[2, 0] * self.a[3, 0]))

            if i in col1:
                theta[2, i] = theta3
            else:
                theta[2, i] = -theta3

        # theta2
        for i in range(8):
            T01 = rbt.dh_transformation_mod(theta[0, i], self.alpha[0, 0], self.d[0, 0], self.a[0, 0])
            T45 = rbt.dh_transformation_mod(theta[4, i], self.alpha[4, 0], self.d[4, 0], self.a[4, 0])
            T56 = rbt.dh_transformation_mod(theta[5, i], self.alpha[5, 0], self.d[5, 0], self.a[5, 0])

            T46 = T45 @ T56
            T60 = rbt.h_inverse(T06)
            T40 = T46 @ T60
            T41 = T40 @ T01
            T14 = rbt.h_inverse(T41)

            p14x = T14[0, 3]
            p14z = T14[2, 3]

            p14xz = np.sqrt(p14x**2 + p14z**2)

            theta2_term1 = np.arctan2(-p14z, -p14x)
            theta2_term2 = np.arcsin((-self.a[3, 0] * np.sin(theta[2, i])) / p14xz)

            theta2 = theta2_term1 - theta2_term2

            theta[1, i] = theta2

        # theta 4
        for i in range(8):
            T01 = rbt.dh_transformation_mod(theta[0, i], self.alpha[0, 0], self.d[0, 0], self.a[0, 0])
            T12 = rbt.dh_transformation_mod(theta[1, i], self.alpha[1, 0], self.d[1, 0], self.a[1, 0])
            T23 = rbt.dh_transformation_mod(theta[2, i], self.alpha[2, 0], self.d[2, 0], self.a[2, 0])
            T03 = T01 @ T12 @ T23

            T45 = rbt.dh_transformation_mod(theta[4, i], self.alpha[4, 0], self.d[4, 0], self.a[4, 0])
            T56 = rbt.dh_transformation_mod(theta[5, i], self.alpha[5, 0], self.d[5, 0], self.a[5, 0])
            T46 = T45 @ T56

            T30 = rbt.h_inverse(T03)
            T64 = rbt.h_inverse(T46)
            T34 = T30 @ T06 @ T64

            Xy34 = T34[1, 0]
            Xx34 = T34[0, 0]
            theta4 = np.arctan2(Xy34, Xx34)
            theta[3, i] = theta4

        return theta

    def plot_arm(self, theta, plt_basis=False, ax=None):
        A1 = rbt.dh_transformation_mod(theta[0, 0], self.alpha[0, 0], self.d[0, 0], self.a[0, 0])
        A2 = rbt.dh_transformation_mod(theta[1, 0], self.alpha[1, 0], self.d[1, 0], self.a[1, 0])
        A3 = rbt.dh_transformation_mod(theta[2, 0], self.alpha[2, 0], self.d[2, 0], self.a[2, 0])
        A4 = rbt.dh_transformation_mod(theta[3, 0], self.alpha[3, 0], self.d[3, 0], self.a[3, 0])
        A5 = rbt.dh_transformation_mod(theta[4, 0], self.alpha[4, 0], self.d[4, 0], self.a[4, 0])
        A6 = rbt.dh_transformation_mod(theta[5, 0], self.alpha[5, 0], self.d[5, 0], self.a[5, 0])

        T01 = A1
        T02 = A1 @ A2
        T03 = A1 @ A2 @ A3
        T04 = A1 @ A2 @ A3 @ A4
        T05 = A1 @ A2 @ A3 @ A4 @ A5
        T06 = A1 @ A2 @ A3 @ A4 @ A5 @ A6

        if plt_basis:
            # plot basic axis
            ax.plot3D([0, 0.5], [0, 0], [0, 0], 'red', linewidth=4)
            ax.plot3D([0, 0], [0, 0.5], [0, 0], 'purple', linewidth=4)
            ax.plot3D([0, 0], [0, 0], [0, 0.5], 'gray', linewidth=4)

        ax.plot3D([0, T01[0, 3]], [0, T01[1, 3]], [0, T01[2, 3]], 'blue', linewidth=8)
        ax.plot3D([T01[0, 3], T02[0, 3]], [T01[1, 3], T02[1, 3]], [T01[2, 3], T02[2, 3]], 'orange', linewidth=8)
        ax.plot3D([T02[0, 3], T03[0, 3]], [T02[1, 3], T03[1, 3]], [T02[2, 3], T03[2, 3]], 'brown', linewidth=8)
        ax.plot3D([T03[0, 3], T04[0, 3]], [T03[1, 3], T04[1, 3]], [T03[2, 3], T04[2, 3]], 'pink', linewidth=8)
        ax.plot3D([T04[0, 3], T05[0, 3]], [T04[1, 3], T05[1, 3]], [T04[2, 3], T05[2, 3]], 'green', linewidth=8)
        ax.plot3D([T05[0, 3], T06[0, 3]], [T05[1, 3], T06[1, 3]], [T05[2, 3], T06[2, 3]], 'cyan', linewidth=8)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')