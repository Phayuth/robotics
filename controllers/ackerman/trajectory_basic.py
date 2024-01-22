import os
import sys
wd = os.path.abspath(os.getcwd())
sys.path.append(str(wd))

import numpy as np
from spatial_geometry.utils import Utilities


class AckermanTrajectoryBasicController:
    """
    [Summary] : Trajectory Basic Controller based on desired point.

    [Method] :

    - Required only Desired Pose(x,y).
    - Orientation is arbitrary.

    """

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
        v = np.sqrt(e[0, 0]**2 + e[1, 0]**2) * self.Kv
        alpha = e[2, 0] * self.Kphi

        if self.upgradedControl:
            # If e[2, 0] is not on the [-pi/2, pi/2], +/- pi should be added
            # to e[2, 0], and negative velocity should be commanded
            v = v * np.sign(np.cos(e[2, 0]))      # Changing sign of v if necessary
            e[2, 0] = np.arctan(np.tan(e[2, 0]))  # Mapped to the [-pi/2, pi/2] interval
            alpha = e[2, 0] * self.Kphi           # Orientation control (upgraded)

        return np.array([[v], [alpha]])


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from robot.mobile.ackerman import AckermanSteer
    from simulator.integrator_euler import EulerNumericalIntegrator
    from trajectory_generator.traj_primitive import fig_of_8

    # create robot and controller
    robot = AckermanSteer(wheelRadius=0, baseLength=0.1)
    controller = AckermanTrajectoryBasicController(robot)

    # simulator
    def dynamic(currentPose, input):
        return robot.forward_external_kinematic(input, currentPose[2, 0])

    def desired(currentPose, time):
        xRef, yRef, _, _, _, _, _, _ = fig_of_8(time)
        phiRef = np.arctan2((yRef - currentPose[1, 0]), (xRef - currentPose[0, 0]))
        return np.array([[xRef], [yRef], [phiRef]])

    def control(currentPose, desiredPose):
        return robot.physical_limit(controller.kinematic_control(currentPose, desiredPose))

    q0 = np.array([[1.1], [0.8], [0]])
    tSpan = (0, 50)
    dt = 0.01
    intg = EulerNumericalIntegrator(dynamic, control, desired, q0, tSpan, dt)
    timeSteps, states, desireds, controls = intg.simulation()

    plt.plot(states[0, :], states[1, :])
    plt.show()