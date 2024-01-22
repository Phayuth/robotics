import os
import sys
wd = os.path.abspath(os.getcwd())
sys.path.append(str(wd))

import numpy as np
from spatial_geometry.spatial_transformation import RigidBodyTransformation as rbt


class Puma560:
    """
    Puma560 - 6DOF Robitic Manipulator. Utilize Modified DH Parameter by J.J. Craig.

    """

    def __init__(self) -> None:
        self.alpha = np.array([[0], [-np.pi / 2], [0], [-np.pi / 2], [np.pi / 2], [-np.pi / 2]])
        self.a = np.array([[0], [0], [0.4318], [-0.0203], [0], [0]])
        self.d = np.array([[0], [0], [-0.0934], [0.4331], [0], [0]])

    def forward_kinematic(self, theta, return_full_H=False, return_each_H=False):
        T01 = rbt.dh_transformation_mod(theta[0, 0], self.alpha[0, 0], self.d[0, 0], self.a[0, 0])
        T12 = rbt.dh_transformation_mod(theta[1, 0], self.alpha[1, 0], self.d[1, 0], self.a[1, 0])
        T23 = rbt.dh_transformation_mod(theta[2, 0], self.alpha[2, 0], self.d[2, 0], self.a[2, 0])
        T34 = rbt.dh_transformation_mod(theta[3, 0], self.alpha[3, 0], self.d[3, 0], self.a[3, 0])
        T45 = rbt.dh_transformation_mod(theta[4, 0], self.alpha[4, 0], self.d[4, 0], self.a[4, 0])
        T56 = rbt.dh_transformation_mod(theta[5, 0], self.alpha[5, 0], self.d[5, 0], self.a[5, 0])

        T06 = T01 @ T12 @ T23 @ T34 @ T45 @ T56

        if return_full_H:  # option to return transformation from base to end effector
            return T06
        if return_each_H:  # option to return all transformation
            return T01, T12, T23, T34, T45, T56

    def inverse_kinematic_geometry(self, goal_desired):
        T06 = goal_desired
        px, py, pz = T06[0, 3], T06[1, 3], T06[2, 3]

        d3 = self.d[2, 0]
        d4 = self.d[3, 0]
        a2 = self.a[2, 0]
        a3 = self.a[3, 0]

        theta = np.zeros((6, 8))

        # theta1
        theta1_1 = np.arctan2(py, px) - np.arctan2(d3, np.sqrt(px**2 + py**2 - d3**2))
        theta1_2 = np.arctan2(py, px) - np.arctan2(d3, -np.sqrt(px**2 + py**2 - d3**2))
        theta[0, 0], theta[0, 1], theta[0, 2], theta[0, 3] = theta1_1, theta1_1, theta1_1, theta1_1
        theta[0, 4], theta[0, 5], theta[0, 6], theta[0, 7] = theta1_2, theta1_2, theta1_2, theta1_2

        # theta3
        K = (px**2 + py**2 + pz**2 - a2**2 - a3**2 - d3**2 - d4**2) / (2 * a2)
        theta3_1 = np.arctan2(a3, d4) - np.arctan2(K, np.sqrt(a3**2 + d4**2 - K**2))
        theta3_2 = np.arctan2(a3, d4) - np.arctan2(K, -np.sqrt(a3**2 + d4**2 - K**2))
        theta[2, 0], theta[2, 1], theta[2, 2], theta[2, 3] = theta3_1, theta3_1, theta3_2, theta3_2
        theta[2, 4], theta[2, 5], theta[2, 6], theta[2, 7] = theta3_1, theta3_1, theta3_2, theta3_2

        # theta2
        for i in range(8):
            theta3 = theta[2, i]
            theta1 = theta[0, i]
            theta23 = np.arctan2(((-a3 - a2 * np.cos(theta3)) * pz + (np.cos(theta1) * px + np.sin(theta1) * py) * (a2 * np.sin(theta3) - d4)),
                                 (a2 * np.sin(theta3) - d4) * pz + (a3 + a2 * np.cos(theta3) * (np.cos(theta1) * px + np.sin(theta1) * py)))
            theta2 = theta23 - theta3
            theta[1, i] = theta2

        # theta 5
        for i in range(8):
            theta1 = theta[0, i]
            theta2 = theta[1, i]
            theta3 = theta[2, i]
            T01 = rbt.dh_transformation_mod(theta1, self.alpha[0, 0], self.d[0, 0], self.a[0, 0])
            T12 = rbt.dh_transformation_mod(theta2, self.alpha[1, 0], self.d[1, 0], self.a[1, 0])
            T23 = rbt.dh_transformation_mod(theta3, self.alpha[2, 0], self.d[2, 0], self.a[2, 0])
            T03 = T01 @ T12 @ T23
            Rprime = rbt.hinverse(T03) @ T06

            r11prm, r12prm, r13prm, pxprm = Rprime[0, 0], Rprime[0, 1], Rprime[0, 2], Rprime[0, 3]
            r21prm, r22prm, r23prm, pyprm = Rprime[1, 0], Rprime[1, 1], Rprime[1, 2], Rprime[1, 3]
            r31prm, r32prm, r33prm, pzprm = Rprime[2, 0], Rprime[2, 1], Rprime[2, 2], Rprime[2, 3]

            sintheta5 = [np.sqrt(r13prm**2 + r33prm**2), -np.sqrt(r13prm**2 + r33prm**2)]
            costheta5 = r23prm

            if i in [0, 2, 4, 6]:
                theta5_1 = np.arctan2(np.sqrt(r13prm**2 + r33prm**2), r23prm)
                theta[4, i] = theta5_1

                # theta 4 and 6
                if 0 not in sintheta5:
                    theta5 = theta5_1
                    theta_4 = np.arctan2(r33prm / np.sin(theta5), -r13prm / np.sin(theta5))
                    theta_6 = np.arctan2(-r22prm / np.sin(theta5), r21prm / np.sin(theta5))

                    if 0 in sintheta5:
                        if theta5 == 0 and costheta5 == 1:
                            theta_4 = 0  # arbitrary
                            theta_6 = np.arctan2(-r12prm, r11prm) - theta_4

                        if theta5 == np.pi and costheta5 == -1:
                            theta_4 = 0  # arb
                            theta_6 = np.arctan2(r12prm, -r11prm) + theta_4

            elif i in [1, 3, 5, 7]:
                theta5_2 = np.arctan2(-np.sqrt(r13prm**2 + r33prm**2), r23prm)
                theta[4, i] = theta5_2

                # theta 4 and 6
                if 0 not in sintheta5:
                    theta5 = theta5_2
                    theta_4 = np.arctan2(r33prm / np.sin(theta5), -r13prm / np.sin(theta5))
                    theta_6 = np.arctan2(-r22prm / np.sin(theta5), r21prm / np.sin(theta5))

                    if 0 in sintheta5:
                        if theta5 == 0 and costheta5 == 1:
                            theta_4 = 0  # arbitrary
                            theta_6 = np.arctan2(-r12prm, r11prm) - theta_4

                        if theta5 == np.pi and costheta5 == -1:
                            theta_4 = 0  # arb
                            theta_6 = np.arctan2(r12prm, -r11prm) + theta_4

            theta[3, i] = theta_4
            theta[5, i] = theta_6

        return theta


if __name__ == "__main__":
    robot = Puma560()
    thetaOriginal = np.array([2, 1, 0.5, 2, 1, 0]).reshape(6, 1)
    TForwardOriginal = robot.forward_kinematic(thetaOriginal, return_full_H=True); print("> TForwardOriginal: \n", TForwardOriginal)
    thetaIk = robot.inverse_kinematic_geometry(TForwardOriginal); print("> thetaIk: \n", thetaIk)

    for i in range(8):
        possibleTheta = thetaIk[:, i].reshape(6, 1)
        TAfterIk = robot.forward_kinematic(possibleTheta, return_full_H=True)
        print(f"> Solution Number {i+1}: {thetaIk[:,i]}",)
        print(TAfterIk)