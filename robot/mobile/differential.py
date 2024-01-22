import os
import sys
wd = os.path.abspath(os.getcwd())
sys.path.append(str(wd))

import numpy as np
from spatial_geometry.spatial_transformation import RigidBodyTransformation as rbt


class DifferentialDrive:

    def __init__(self, wheelRadius, baseLength, baseWidth) -> None:
        # kinematic properties | dynamic properties
        self.r = wheelRadius  # m
        self.L = baseLength   # m
        self.mass = 0.75      # mass kg
        self.I = 0.001        # moment inertial kg*m^2
        self.d = 0.01         # wheel center to robot center of mass

        # view
        self.color = "olive"

        # collision assumming robot as a circle
        self.W = baseWidth    # m
        self.collisionOffset = 0
        self.collisionR = np.sqrt((self.L/2)**2 +(self.W/2)**2) + self.collisionOffset

    def forward_internal_kinematic(self, wheelVelo):  # wheelVelo is rotation omega
        rightWheelVelo = wheelVelo[0, 0]
        leftWheelVelo = wheelVelo[1, 0]
        linearVelo = (self.r / 2) * (rightWheelVelo + leftWheelVelo)
        angularVelo = (self.r / self.L) * (rightWheelVelo - leftWheelVelo)
        return np.array([[linearVelo], [angularVelo]])

    def forward_external_kinematic(self, bodyVelo, theta):  # find xdot, ydot, thetadot from V and omega
        return np.array([[bodyVelo[0, 0] * np.cos(theta)],
                         [bodyVelo[0, 0] * np.sin(theta)],
                         [bodyVelo[1, 0]                ]])

    def inverse_internal_kinematic(self, bodyVelo):
        linearVelo = bodyVelo[0, 0]
        angularVelo = bodyVelo[1, 0]
        rightWheelVelo = (2*linearVelo + angularVelo * self.L) / (2 * self.r)
        leftWheelVelo  = (2*linearVelo - angularVelo * self.L) / (2 * self.r)
        return np.array([[rightWheelVelo], [leftWheelVelo]])

    def forward_internal_dynamic(self, torque):
        E = np.array([[        1 / self.mass * self.r,          1 / self.mass * self.r],
                      [self.L / (2 * self.r * self.I), -self.L / (2 * self.r * self.I)]])
        acceleration = E @ torque
        return acceleration # linear acc, angular acc

    def forward_external_dynamic(self, phi, v, w, torque): # (phi = theta = heading), linear velo, angular velo, torque
        F = np.array([[                        v * np.cos(phi) - self.d * w * np.sin(phi)],
                      [                        v * np.sin(phi) + self.d * w * np.cos(phi)],
                      [                                                                 w],
                      [                                                   self.d * (w**2)],
                      [-(self.d * w * v * self.mass) / (self.mass * (self.d**2) + self.I)]])

        temp1 = 1 / (self.mass*self.r)
        temp2 = self.L / (2 * self.r * (self.mass * (self.d**2) + self.I))
        G = np.array([[    0,      0],
                      [    0,      0],
                      [    0,      0],
                      [temp1,  temp1],
                      [temp2, -temp2]])
        return F + G @ torque

    def inverse_external_dynamic(self, v, w, dv, dw): # linear velo, angular velo, linear acc, angular acc
        firstTerm = (self.r * (dv*self.mass - self.d*w*self.mass*w)) / 2
        lastTerm = (self.r * (dw * (self.mass * (self.d**2) + self.I) + self.d*w*self.mass*v)) / self.L
        return np.array([[firstTerm + lastTerm],
                         [firstTerm - lastTerm]])

    def robot_link(self, pose, return_vertices=False):
        xPose = pose[0, 0]
        yPose = pose[1, 0]
        tPose = pose[2, 0]

        v1 = np.array([[-self.W / 2], [-self.L / 2]])
        v2 = np.array([[ self.W / 2], [-self.L / 2]])
        v3 = np.array([[ self.W / 2], [ self.L / 2]])
        v4 = np.array([[-self.W / 2], [ self.L / 2]])

        v1 = rbt.rot2d(tPose) @ v1 + np.array([[xPose], [yPose]])
        v2 = rbt.rot2d(tPose) @ v2 + np.array([[xPose], [yPose]])
        v3 = rbt.rot2d(tPose) @ v3 + np.array([[xPose], [yPose]])
        v4 = rbt.rot2d(tPose) @ v4 + np.array([[xPose], [yPose]])

        if return_vertices:
            return v1, v2, v3, v4
        else:
            return [[  xPose,   yPose],
                    [v1[0,0], v1[1,0]],
                    [v2[0,0], v2[1,0]],
                    [v3[0,0], v3[1,0]],
                    [v4[0,0], v4[1,0]]]

    def plot_robot(self, pose, axis):
        v1, v2, v3, v4 = self.robot_link(pose, return_vertices=True)

        axis.plot([v1[0, 0], v2[0, 0]], [v1[1, 0], v2[1, 0]], c=self.color)
        axis.plot([v2[0, 0], v3[0, 0]], [v2[1, 0], v3[1, 0]], c=self.color)
        axis.plot([v3[0, 0], v4[0, 0]], [v3[1, 0], v4[1, 0]], c=self.color)
        axis.plot([v4[0, 0], v1[0, 0]], [v4[1, 0], v1[1, 0]], c=self.color)

        axis.plot([pose[0, 0], v2[0, 0]], [pose[1, 0], v2[1, 0]], c=self.color)
        axis.plot([pose[0, 0], v3[0, 0]], [pose[1, 0], v3[1, 0]], c=self.color)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from simulator.integrator_euler import EulerNumericalIntegrator
    from trajectory_generator.traj_primitive import fig_of_8_with_acceleration

    robot = DifferentialDrive(wheelRadius=0.024, baseLength=0.075, baseWidth=0.075)
    pose = np.array([0, 0, 0]).reshape(3, 1)

    fig, ax = plt.subplots(1, 1)
    ax.grid(True)
    ax.set_aspect("equal")
    robot.plot_robot(pose, ax)
    plt.show()

    # simulator
    def dynamic(currentPose, input):
        phi = currentPose[2, 0]
        v = currentPose[3, 0]
        w = currentPose[4, 0]
        return robot.forward_external_dynamic(phi, v, w, input)

    def desired(currentPose, time):
        xRef, yRef, dxRef, dyRef, ddxRef, ddyRef, dddxRef, dddyRef, vRef, wRef, dvRef, dwRef = fig_of_8_with_acceleration(time)
        return np.array([[vRef], [wRef], [dvRef], [dwRef]])

    def control(currentPose, desiredPose):
        v = desiredPose[0, 0]
        w = desiredPose[1, 0]
        dv = desiredPose[2, 0]
        dw = desiredPose[3, 0]
        return robot.inverse_external_dynamic(v, w, dv, dw)

    xRef, yRef, dxRef, dyRef, ddxRef, ddyRef, dddxRef, dddyRef, vRef, wRef, dvRef, dwRef = fig_of_8_with_acceleration(t=0)
    q0 = np.array([[xRef], [yRef], [np.arctan2(dyRef, dxRef)], [vRef], [wRef]])
    tSpan = (0, 100)
    dt = 0.01
    intg = EulerNumericalIntegrator(dynamic, control, desired, q0, tSpan, dt)
    timeSteps, states, desireds, controls = intg.simulation()

    plt.plot(states[0, :], states[1, :])
    plt.show()