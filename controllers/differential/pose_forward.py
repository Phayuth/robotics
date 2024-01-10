"""
Pose Forward Controller for Differential Drive Mobile Robot
Reference : WMR book P72 ex3.3 Control to Reference Pose Using an Intermediate Point
the final orientation is defined

"""
import numpy as np
from spatial_geometry.utils import Utilities


class DifferentialDrivePoseForwardController:

    def __init__(self, robot) -> None:
        self.robot = robot

        # controller constants for tuning
        self.K1 = 0.8
        self.K2 = 5.0

        # intermediate pose circle
        self.r = 0.5  # Distance parameter for the intermediate point
        self.dTol = 0.05  # Tolerance distance (to the intermediate point) for switch
        self.state = False  # State: 0 - go to intermediate point, 1 - go to reference point

    def kinematic_control(self, currentPose, referencePose):
        xT = referencePose[0, 0] - self.r * np.cos(referencePose[2, 0])
        yT = referencePose[0, 0] - self.r * np.sin(referencePose[2, 0])

        D = np.sqrt((referencePose[0, 0] - currentPose[0, 0])**2 + (referencePose[1, 0] - currentPose[1, 0])**2)
        if D < self.dTol:
            vc = 0
            wc = 0
        else:
            if self.state is False:
                d = np.sqrt((xT - currentPose[0, 0])**2 + (yT - currentPose[1, 0])**2)
                if d < self.dTol:
                    self.state = True

                phiT = np.arctan2(yT - currentPose[1, 0], xT - currentPose[0, 0])
                ePhi = phiT - currentPose[2, 0]
            else:
                ePhi = referencePose[2, 0] - currentPose[2, 0]

            ePhi = Utilities.wrap_to_pi(ePhi)

            vc = self.K1 * D
            wc = self.K2 * ePhi

        return np.array([vc, wc]).reshape(2, 1)