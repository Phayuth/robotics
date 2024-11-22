import os
import sys

sys.path.append(str(os.path.abspath(os.getcwd())))

import numpy as np


class DifferentialDriveLinearizeFeedbackController:

    def __init__(self, robot) -> None:
        self.robot = robot

        # controller constants for tuning
        self.K1 = 5.0
        self.K2 = 15.0
        self.K3 = 75.0
        self.K4 = 125.0

        self.dTol = 0.05  # Tolerance distance (to the intermediate point) for switch

    def kinematic_control(self, currentPose, referencePose):
        # return np.array([vc, wc]).reshape(2, 1)
        pass



if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from robot.mobile.differential import DifferentialDrive
    from simulator.integrator_euler import EulerNumericalIntegrator

    # create robot and controller
    robot = DifferentialDrive(wheelRadius=0, baseLength=0.3, baseWidth=0.3)
    controller = DifferentialDriveLinearizeFeedbackController(robot=robot)

    # simulator
    def dynamic(currentPose, input):
        return robot.forward_external_kinematic(input, currentPose[2, 0])

    def desired(currentPose, time):
        return np.array([2.0, 2.0, np.pi / 2]).reshape(3, 1)

    def control(currentPose, desiredPose):
        return controller.kinematic_control(currentPose, desiredPose)

    q0 = np.array([0.0, 0.0, -100 / 180 * np.pi]).reshape(3, 1)
    tSpan = (0, 10)
    dt = 0.01
    intg = EulerNumericalIntegrator(dynamic, control, desired, q0, tSpan, dt)
    timeSteps, states, desireds, controls = intg.simulation()

    plt.plot(states[0, :], states[1, :])
    # plt.plot(timeSteps, states[0,:])
    # plt.plot(timeSteps, states[1,:])
    # plt.plot(timeSteps, states[2,:])
    plt.show()
