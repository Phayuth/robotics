"""
Kinematic Trajectory Error Tracking Model controller for Linear Differential Drive Mobile Robot
Reference : WMR book P100-105 ex3.9

"""
import numpy as np
from spatial_geometry.utils import Utilities


class DifferentialDriveErrorLinearTrajectoryController:

    def __init__(self, robot) -> None:
        self.robot = robot

        # kinematic controller constants for tuning
        self.zeta = 0.9
        self.g = 85

    def kinematic_control(self, currentPose, referencePose, referenceLinearVelo, referenceAngularVelo):
        T = np.array([[np.cos(currentPose[2, 0]), np.sin(currentPose[2, 0]), 0], [-np.sin(currentPose[2, 0]), np.cos(currentPose[2, 0]), 0], [0, 0, 1]])
        qe = T @ (referencePose-currentPose)

        ePhi = Utilities.wrap_to_pi(qe[2, 0])
        Kx = 2 * self.zeta * np.sqrt((referenceAngularVelo**2) + self.g * (referenceLinearVelo**2))
        Kphi = Kx
        Ky = self.g * referenceLinearVelo
        # Gains can also be constant e.g.: Kx = Kphi = 3; Ky = 30

        # Control: feedforward and feedback
        vc = referenceLinearVelo * np.cos(ePhi) + Kx*qe[0, 0]
        wc = referenceAngularVelo + Ky*qe[1, 0] + Kphi*ePhi

        return np.array([vc, wc]).reshape(2, 1)