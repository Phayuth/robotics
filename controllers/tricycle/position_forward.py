"""
Pose Forward Controller for Tricycle Rear-Powered Mobile Robot
Reference : WMR book P68 ex3.1
In this case is a basic position, the final orientation can be arbitrary

"""
import numpy as np


class TricycleDrivePositionForwardController:

    def __init__(self, robot):
        self.robot = robot

        # controller constants for tuning
        self.K1 = 0.3
        self.K2 = 0.2

    def kinematic_control(self, currentPose, referencePose):
        e = referencePose - currentPose

        V = self.K1 * np.sqrt((e[0, 0]**2) + (e[1, 0]**2))
        O = self.K2 * e[2, 0]

        controlValue = np.array([[V], [O]])
        controlValue = self.robot.physical_limit(controlValue)

        return np.array([[V], [O]])
