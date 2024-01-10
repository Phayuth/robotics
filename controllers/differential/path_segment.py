"""
Path Segment Control for Differential Drive Mobile Robot
Reference : WMR book P83 ex3.6

"""
import numpy as np
from spatial_geometry.utils import Utilities


class DifferentialDrivePathSegmentTrackingController:

    def __init__(self, robot, referenceSegment) -> None:
        self.robot = robot
        self.referenceSegment = referenceSegment

        # kinematic controller constants for tuning
        self.cte1 = 0.9  # 0.4 good
        self.cte2 = 3  # 3 good

        # path segment id
        self.i = 0

    def kinematic_control(self, currentPose):
        #Reference segment determination
        dx = np.array(self.referenceSegment[self.i + 1] - self.referenceSegment[self.i])[0]  # correct take x each line segment to find dx
        dy = np.array(self.referenceSegment[self.i + 1] - self.referenceSegment[self.i])[1]  # correct take y each line segment to find dy

        V = np.array([[dx], [dy]])
        Vn = np.array([[dy], [-dx]])

        rx = currentPose[0, 0] - self.referenceSegment[self.i, 0]
        ry = currentPose[1, 0] - self.referenceSegment[self.i, 1]
        r = np.array([[rx], [ry]])
        u = (np.transpose(V) @ r) / (np.transpose(V) @ V)

        if u > 1 and self.i < np.shape(self.referenceSegment)[0] - 1:
            self.i = self.i + 1
            dx = np.array(self.referenceSegment[self.i + 1] - self.referenceSegment[self.i])[0]
            dy = np.array(self.referenceSegment[self.i + 1] - self.referenceSegment[self.i])[1]

            V = np.array([[dx], [dy]])
            Vn = np.array([[dy], [-dx]])

            rx = currentPose[0, 0] - self.referenceSegment[self.i, 0]
            ry = currentPose[1, 0] - self.referenceSegment[self.i, 1]
            r = np.array([[rx], [ry]])

        dn = (np.transpose(Vn) @ r) / (np.transpose(Vn) @ Vn)

        phiLin = np.arctan2(V[1, 0], V[0, 0])
        phiRot = np.arctan(5 * dn)
        phiRef = phiLin + phiRot
        phiRef = Utilities.wrap_to_pi(phiRef)

        ephi = phiRef - currentPose[2, 0]
        ephi = Utilities.wrap_to_pi(ephi)

        vc = self.cte1 * np.cos(ephi)
        wc = self.cte2 * ephi

        return np.array([vc, wc]).reshape(2, 1)