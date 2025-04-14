import numpy as np


class DifferentialDriveBackSteppingTrajectoryController:

    def __init__(self, robot) -> None:
        self.robot = robot

        # kinematic controller constants for tuning
        self.k1 = 10
        self.k2 = 5
        self.k3 = 4

        # dynamic controller constants for tuning
        self.ka = 100
        self.kb = 3000

    def kinematic_control(self, currentPose, referencePose, referenceLinearVelo, referenceAngularVelo):
        T = np.array([[ np.cos(currentPose[2, 0]), np.sin(currentPose[2, 0]), 0],
                      [-np.sin(currentPose[2, 0]), np.cos(currentPose[2, 0]), 0],
                      [                         0,                         0, 1]])
        qe = T @ (referencePose-currentPose)
        vc = referenceLinearVelo * np.cos(qe[2, 0]) + self.k1 * qe[0, 0]
        wc = referenceAngularVelo + self.k2 * referenceLinearVelo * qe[1, 0] + self.k3 * np.sin(qe[2, 0])

        return np.array([[vc], [wc]])

    def dynamic_control(self, vcurrent, wcurrent, vref, wref, vdotref, wdotref):  # is not fully correct yet
        mass = self.robot.mass
        radius = self.robot.r
        length = self.robot.L
        Inertial = self.robot.I
        z1 = vref - vcurrent  # x1 = v
        z2 = wref - wcurrent  # x2 = omega

        tua1c = 1 / 2 * ((mass * radius * (vdotref + self.ka * z1)) + ((2*radius*Inertial/length) * (wdotref + self.kb * z2)))
        tua2c = 1 / 2 * ((mass * radius * (vdotref + self.ka * z1)) - ((2*radius*Inertial/length) * (wdotref + self.kb * z2)))

        return np.array([[tua1c], [tua2c]])


if __name__=="__main__":
    import os
    import sys
    wd = os.path.abspath(os.getcwd())
    sys.path.append(str(wd))

    import matplotlib.pyplot as plt
    from robot.mobile.differential import DifferentialDrive
    from simulator.integrator_euler import EulerNumericalIntegrator
    from trajectory_generator.traj_primitive import circle

    # create robot and controller
    robot = DifferentialDrive(wheelRadius=0.03, baseLength=0.3, baseWidth=1)
    controller = DifferentialDriveBackSteppingTrajectoryController(robot=robot)

    # simulator
    def dynamic(currentPose, input):
        return robot.forward_external_kinematic(input, currentPose[2,0])

    def desired(currentPose, time):
        xRef, yRef, vr, wr, ydot, xdot, vdotref, wdotref = circle(time)
        return np.array([[xRef], [yRef], [vr], [wr], [xdot], [ydot]])

    def control(currentPose, desiredPose):
        xRef = desiredPose[0,0]
        yRef = desiredPose[1,0]
        vr = desiredPose[2,0]
        wr = desiredPose[3,0]
        xdot = desiredPose[4,0]
        ydot = desiredPose[5,0]
        thetaRef = np.arctan2(ydot, xdot)
        qr = np.array([[xRef], [yRef], [thetaRef]])
        return controller.kinematic_control(currentPose, qr, vr, wr)

    q0 = np.array([[6], [0], [0]])
    tSpan = (0, 50)
    dt = 0.01
    intg = EulerNumericalIntegrator(dynamic, control, desired, q0, tSpan, dt)
    timeSteps, states, desireds, controls = intg.simulation()

    plt.plot(states[0,:], states[1,:])
    # plt.plot(timeSteps, states[0,:])
    # plt.plot(timeSteps, states[1,:])
    # plt.plot(timeSteps, states[2,:])
    plt.show()