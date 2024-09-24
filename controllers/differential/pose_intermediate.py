import os
import sys

sys.path.append(str(os.path.abspath(os.getcwd())))

import numpy as np
from spatial_geometry.utils import Utils


class DifferentialDriveIntermediatePoseController:
    """
    [Summary] : Pose Intermediate Controller provide an intermediate point where trajectory is shape in a way desired orientation is obtained.

    """

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

        D = np.sqrt((referencePose[0, 0] - currentPose[0, 0]) ** 2 + (referencePose[1, 0] - currentPose[1, 0]) ** 2)
        if D < self.dTol:
            vc = 0
            wc = 0
        else:
            if self.state is False:
                d = np.sqrt((xT - currentPose[0, 0]) ** 2 + (yT - currentPose[1, 0]) ** 2)
                if d < self.dTol:
                    self.state = True

                phiT = np.arctan2(yT - currentPose[1, 0], xT - currentPose[0, 0])
                ePhi = phiT - currentPose[2, 0]
            else:
                ePhi = referencePose[2, 0] - currentPose[2, 0]

            ePhi = Utils.wrap_to_pi(ePhi)

            vc = self.K1 * D
            wc = self.K2 * ePhi

        return np.array([vc, wc]).reshape(2, 1)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from robot.mobile.differential import DifferentialDrive
    from simulator.integrator_euler import EulerNumericalIntegrator

    # create robot and controller
    robot = DifferentialDrive(wheelRadius=0, baseLength=0.3, baseWidth=0.3)
    controller = DifferentialDriveIntermediatePoseController(robot=robot)

    # simulator
    def dynamic(currentPose, input):
        return robot.forward_external_kinematic(input, currentPose[2, 0])

    def desired(currentPose, time):
        return np.array([2.0, 2.0, np.pi / 2]).reshape(3, 1)

    def control(currentPose, desiredPose):
        return controller.kinematic_control(currentPose, desiredPose)

    q0 = np.array([1.0, 0.0, -100 / 180 * np.pi]).reshape(3, 1)
    tSpan = (0, 10)
    dt = 0.01
    intg = EulerNumericalIntegrator(dynamic, control, desired, q0, tSpan, dt)
    timeSteps, states, desireds, controls = intg.simulation()

    plt.plot(states[0, :], states[1, :])
    # plt.plot(timeSteps, states[0,:])
    # plt.plot(timeSteps, states[1,:])
    # plt.plot(timeSteps, states[2,:])
    plt.show()
