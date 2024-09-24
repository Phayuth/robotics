import os
import sys

sys.path.append(str(os.path.abspath(os.getcwd())))

import numpy as np
from spatial_geometry.utils import Utils


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
        # Reference segment determination
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
        phiRef = Utils.wrap_to_pi(phiRef)

        ephi = phiRef - currentPose[2, 0]
        ephi = Utils.wrap_to_pi(ephi)

        vc = self.cte1 * np.cos(ephi)
        wc = self.cte2 * ephi

        return np.array([vc, wc]).reshape(2, 1)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from robot.mobile.differential import DifferentialDrive
    from simulator.integrator_euler import EulerNumericalIntegrator

    # create robot and controller
    qref = np.array([[3, 0], [6, 4], [3, 4], [3, 1], [0, 3], [0, 5], [-1, 7], [-5, 0], [-6, 3]])
    robot = DifferentialDrive(wheelRadius=0.03, baseLength=0.3, baseWidth=0.3)
    controller = DifferentialDrivePathSegmentTrackingController(robot=robot, referenceSegment=qref)

    # simulator
    def dynamic(currentPose, input):
        return robot.forward_external_kinematic(input, currentPose[2, 0])

    def desired(currentPose, time):
        return np.array([0.0, 0.0, 0]).reshape(3, 1)  # isn't needed in this type of control mode

    def control(currentPose, desiredPose):
        return controller.kinematic_control(currentPose)

    q0 = np.array([[5], [1], [0.6 * np.pi]])
    tSpan = (0, 32)
    dt = 0.01
    intg = EulerNumericalIntegrator(dynamic, control, desired, q0, tSpan, dt)
    timeSteps, states, desireds, controls = intg.simulation()

    plt.grid(True)
    plt.plot(qref[:, 0], qref[:, 1])
    plt.plot(states[0, :], states[1, :])
    # plt.plot(timeSteps, states[0,:])
    # plt.plot(timeSteps, states[1,:])
    # plt.plot(timeSteps, states[2,:])
    plt.show()
