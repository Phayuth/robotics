"""
Title : Kinematic Trajectory Basic Controller for Ackerman Steering Car

Description :
    the trajectory basic provide only x and y. (no infomation of velocity and accerlation)

Reference : WMR book P88 ex3.7

"""
import numpy as np
from spatial_geometry.utils import Utilities


class AckermanTrajectoryBasicController:

    def __init__(self, robot) -> None:
        self.robot = robot

        # Control Gain
        self.Kphi = 2
        self.Kv = 5

        self.upgradedControl = True

    def kinematic_control(self, currentPose, referencePose):
        e = referencePose - currentPose
        e[2, 0] = Utilities.wrap_to_pi(e[2, 0])

        # control basic
        v = np.sqrt(e[0, 0]**2 + e[1, 0]**2) * self.Kv  # forward motion control
        alpha = e[2, 0] * self.Kphi  # orientation control

        if self.upgradedControl:
            # If e[2, 0] is not on the [-pi/2, pi/2], +/- pi should be added
            # to e[2, 0], and negative velocity should be commanded
            v = v * np.sign(np.cos(e[2, 0]))  # Changing sign of v if necessary
            e[2, 0] = np.arctan(np.tan(e[2, 0]))  # Mapped to the [-pi/2, pi/2] interval
            alpha = e[2, 0] * self.Kphi  # Orientation control (upgraded)

        return np.array([[v], [alpha]])