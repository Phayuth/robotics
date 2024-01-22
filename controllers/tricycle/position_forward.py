import numpy as np


class TricycleRearPoweredDrivePositionController:
    """
    [Summary] : Position Controller for Rear Powered provide control for only Position. Orientation can be arbitrary.

    """

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


if __name__=="__main__":
    import os
    import sys
    wd = os.path.abspath(os.getcwd())
    sys.path.append(str(wd))

    import matplotlib.pyplot as plt
    from robot.mobile.tricycle import TricycleDrive
    from simulator.integrator_euler import EulerNumericalIntegrator

    # create robot and controller
    robot = TricycleDrive(wheelRadius=0, baseLength=0.1)
    controller = TricycleRearPoweredDrivePositionController(robot=robot)

    # simulator
    def dynamic(currentPose, input):
        return robot.forward_external_kinematic(input, currentPose[2,0])

    def desired(currentPose, time):
        return np.array([[4.0], [4.0]])

    def control(currentPose, desiredPose):
        phiref = np.arctan2((desiredPose[1, 0] - currentPose[1, 0]), (desiredPose[0, 0] - currentPose[0, 0]))
        qRef = np.array([desiredPose[0, 0], desiredPose[1, 0], phiref]).reshape(3, 1)
        return controller.kinematic_control(currentPose, qRef)

    q0 = np.array([1.0, 0.0, -np.pi]).reshape(3, 1)
    tSpan = (0, 50)
    dt = 0.01
    intg = EulerNumericalIntegrator(dynamic, control, desired, q0, tSpan, dt)
    timeSteps, states, desireds, controls = intg.simulation()

    plt.plot(states[0,:], states[1,:])
    plt.show()