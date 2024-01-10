import os
import sys
wd = os.path.abspath(os.getcwd())
sys.path.append(str(wd))

import numpy as np
from spatial_geometry.spatial_transformation import RigidBodyTransformation as rbt


class DifferentialDrive:

    def __init__(self, wheelRadius, baseLength, baseWidth) -> None:
        # kinematic properties
        self.r = wheelRadius  # m
        self.L = baseLength   # m
        self.W = baseWidth    # m

        # dynamic properties
        self.mass = 4  # mass kg
        self.I = 2.5   # inertial kg*m^2

        # view
        self.color = "olive"

        # collision assumming robot as a circle
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
        leftWheelVelo = (2*linearVelo - angularVelo * self.L) / (2 * self.r)
        return np.array([[rightWheelVelo], [leftWheelVelo]])

    def forward_internal_dynamic(self, torque):
        E = np.array([[        1 / self.mass * self.r,           1 / self.mass * self.r],
                      [self.L / (2 * self.r * self.I), -self.L / (2 * self.r * self.I)]])
        acceleration = E @ torque
        return acceleration # linear acc, angular acc

    def robot_link(self, pose, return_vertices=False):
        xPose = pose[0, 0]
        yPose = pose[1, 0]
        tPose = pose[2, 0]

        v1 = np.array([-self.W / 2, -self.L / 2]).reshape(2, 1)
        v2 = np.array([ self.W / 2, -self.L / 2]).reshape(2, 1)
        v3 = np.array([ self.W / 2,  self.L / 2]).reshape(2, 1)
        v4 = np.array([-self.W / 2,  self.L / 2]).reshape(2, 1)

        v1 = rbt.rot2d(tPose) @ v1 + np.array([xPose, yPose]).reshape(2, 1)
        v2 = rbt.rot2d(tPose) @ v2 + np.array([xPose, yPose]).reshape(2, 1)
        v3 = rbt.rot2d(tPose) @ v3 + np.array([xPose, yPose]).reshape(2, 1)
        v4 = rbt.rot2d(tPose) @ v4 + np.array([xPose, yPose]).reshape(2, 1)

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

    robot = DifferentialDrive(wheelRadius=0.1, baseLength=1, baseWidth=1.7)
    pose = np.array([0, 0, 0]).reshape(3, 1)

    fig, ax = plt.subplots(1, 1)
    ax.grid(True)
    ax.set_aspect("equal")
    ax.set_xlim([-10, 10])
    ax.set_ylim([-10, 10])
    robot.plot_robot(pose, ax)
    plt.show()