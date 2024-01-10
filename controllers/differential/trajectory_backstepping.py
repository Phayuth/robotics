"""
Backstepping controller for Differential Drive Mobile Robot
https://ieeexplore.ieee.org/abstract/document/7153286

"""
import numpy as np


class DifferentialDriveBackSteppingTrajectoryController:

    def __init__(self, robot) -> None:
        self.robot = robot

        # kinematic controller constants for tuning
        self.k1 = 10
        self.k2 = 5
        self.k3 = 4

        # dynamic controller constants for tuning
        self.ka = 100
        self.kb = 3000

    def kinematic_control(self, currentPose, referencePose, referenceLinearVelo, referenceAngularVelo):
        T = np.array([[ np.cos(currentPose[2, 0]), np.sin(currentPose[2, 0]), 0],
                      [-np.sin(currentPose[2, 0]), np.cos(currentPose[2, 0]), 0],
                      [                         0,                         0, 1]])
        qe = T @ (referencePose-currentPose)
        vc = referenceLinearVelo * np.cos(qe[2, 0]) + self.k1 * qe[0, 0]
        wc = referenceAngularVelo + self.k2 * referenceLinearVelo * qe[1, 0] * self.k3 * np.sin(qe[2, 0])

        return np.array([[vc], [wc]])

    def dynamic_control(self, vcurrent, wcurrent, vref, wref, vdotref, wdotref):  # is not fully correct yet
        mass = self.robot.mass
        radius = self.robot.r
        length = self.robot.L
        Inertial = self.robot.I
        z1 = vref - vcurrent  # x1 = v
        z2 = wref - wcurrent  # x2 = omega

        tua1c = 1 / 2 * ((mass * radius * (vdotref + self.ka * z1)) + ((2*radius*Inertial/length) * (wdotref + self.kb * z2)))
        tua2c = 1 / 2 * ((mass * radius * (vdotref + self.ka * z1)) - ((2*radius*Inertial/length) * (wdotref + self.kb * z2)))

        return np.array([[tua1c], [tua2c]])