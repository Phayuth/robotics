"""
Pose Basic Controller for Differential Drive Mobile Robot
Reference : WMR book P70 Control to Reference Position
the final orientation is defined
very basic one:
    - go to position first until e is smaller at some small eta
    - then correct orientation

"""
import numpy as np


class DifferentialDrivePoseBasicController:

    def __init__(self, robot) -> None:
        self.robot = robot

        # controller constants for tuning
        self.K1 = 0.8
        self.K2 = 5
        self.Korient = 0.5

        # correction
        self.dTol = 0.01  # Tolerance distance (to the intermediate point) for switch
        self.state = False  # State: 0 - go to position, 1 - go to orientation

    def kinematic_control(self, currentPose, referencePose):
        # controller switch condition
        if np.linalg.norm([(referencePose[0,0] - currentPose[0,0]),(referencePose[1,0] - currentPose[1,0])]) < self.dTol:
            self.state = True

        # position controller
        if not self.state:
            phiref = np.arctan2((referencePose[1, 0] - currentPose[1, 0]), (referencePose[0, 0] - currentPose[0, 0]))
            qRef = np.array([referencePose[0, 0], referencePose[1, 0], phiref]).reshape(3, 1)
            e = qRef - currentPose
            vc = self.K1 * np.sqrt((e[0, 0]**2) + (e[1, 0]**2))
            wc = self.K2 * e[2, 0]

        # orientation controller
        if self.state:
            e = referencePose[2, 0] - currentPose[2, 0]
            vc = 0
            wc = self.Korient * e

        return np.array([[vc], [wc]])


class DifferentialDrivePositionForwardController:

    def __init__(self, robot) -> None:
        self.robot = robot

        # controller constants for tuning
        self.K1 = 0.1
        self.K2 = 0.5

    def kinematic_control(self, currentPose, referencePose):
        e = referencePose - currentPose

        vc = self.K1 * np.sqrt((e[0, 0]**2) + (e[1, 0]**2))
        wc = self.K2 * e[2, 0]

        if abs(vc) > 0.8:
            vc = 0.8 * np.sign(vc)

        return np.array([vc, wc]).reshape(2, 1)