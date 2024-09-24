import os
import sys
sys.path.append(str(os.path.abspath(os.getcwd())))

import numpy as np
from spatial_geometry.spatial_transformation import RigidBodyTransformation as rbt


class BicycleDriveFrontSteering:

    def __init__(self, wheelRadius, baseLength) -> None:
        # kinematic properties
        self.r = wheelRadius  #m
        self.d = baseLength   #m

    def forward_internal_kinematic(self, motionInput):
        omegas = motionInput[0, 0] # front wheel rotation velo (front power)
        alpha = motionInput[1, 0]  # steering angle
        vS = omegas * self.r
        v = vS * np.cos(alpha)
        omega = (vS/self.d) * np.sin(alpha)
        return np.array([v, omega]).reshape(2, 1)

    def forward_external_kinematic(self, bodyVelo, phi):  # find xdot, ydot, thetadot from V and omega, pose measure at rear wheel
        J = np.array([[np.cos(phi), 0],
                      [np.sin(phi), 0],
                      [          0, 1]])
        globalVelo = J @ bodyVelo
        return globalVelo

    def robot_link(self, pose, return_vertices=False):
        xPose = pose[0, 0]
        yPose = pose[1, 0]
        tPose = pose[2, 0]
        aPose = pose[3, 0]

        v1 = np.array([self.d, 0]).reshape(2, 1)
        v2 = rbt.rot2d(aPose) @ np.array([-self.r, 0]).reshape(2, 1) + np.array([self.d, 0]).reshape(2, 1)
        v3 = rbt.rot2d(aPose) @ np.array([ self.r, 0]).reshape(2, 1) + np.array([self.d, 0]).reshape(2, 1)

        v1 = rbt.rot2d(tPose) @ v1 + np.array([xPose, yPose]).reshape(2, 1)
        v2 = rbt.rot2d(tPose) @ v2
        v3 = rbt.rot2d(tPose) @ v3

        if return_vertices:
            return v1, v2, v3
        else:
            return [[  xPose,   yPose],
                    [v1[0,0], v1[1,0]],
                    [v2[0,0], v2[1,0]],
                    [v3[0,0], v3[1,0]]]

    def plot_robot(self, pose, axis):
        v1, v2, v3 = self.robot_link(pose, return_vertices=True)

        axis.plot([v1[0, 0]], [v1[1, 0]], 'ro')
        axis.plot([pose[0, 0], v1[0, 0]], [pose[1, 0], v1[1, 0]], c='blue')
        axis.plot([v2[0, 0], v3[0, 0]], [v2[1, 0], v3[1, 0]], c='red')


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    robot = BicycleDriveFrontSteering(wheelRadius=0.3, baseLength=1)
    pose = np.array([0, 0, 0, 1]).reshape(4, 1)

    fig, ax = plt.subplots(1, 1)
    ax.grid(True)
    ax.set_aspect("equal")
    ax.set_xlim([-10, 10])
    ax.set_ylim([-10, 10])
    robot.plot_robot(pose, ax)
    plt.show()