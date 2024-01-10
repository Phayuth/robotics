"""
Mecanum 4 Wheels Robot
Reference : WMR book P30
"""
import os
import sys
wd = os.path.abspath(os.getcwd())
sys.path.append(str(wd))

import numpy as np
from spatial_geometry.spatial_transformation import RigidBodyTransformation as rbt


class MecanumDrive:

    def __init__(self, wheelRadius, baseLength, baseWidth) -> None:
        # kinematic properties
        self.r = wheelRadius
        self.l = baseLength
        self.d = baseWidth

        # view
        self.color = "olive"

        # collision assumming robot as a circle
        self.collisionOffset = 0
        self.collisionR = np.sqrt((self.l)**2 +(self.d)**2) + self.collisionOffset

    def forward_internal_kinematic(self, wheelVelo):
        Jpseudo = np.array([[                  1,                    1,                   1,                   1],
                           [                  -1,                    1,                  -1,                   1],
                           [-1/(self.l + self.d), -1/(self.l + self.d), 1/(self.l + self.d), 1/(self.l + self.d)]])
        localVelo = ((1/4) * Jpseudo) @ wheelVelo
        return localVelo

    def forward_external_kinematic(self, localVelo, phi):
        globalVelo = self.rotz(phi).T @ localVelo
        return globalVelo

    def inverse_external_kinematic(self, gobalVelo, phi):
        J = np.array([[1, -1, -(self.l + self.d)],
                      [1,  1, -(self.l + self.d)],
                      [1, -1,  (self.l + self.d)],
                      [1,  1,  (self.l + self.d)]])
        wheelVelo = J @ self.rotz(phi) @ gobalVelo
        return wheelVelo

    def rotz(self, theta):
        R = np.array([[ np.cos(theta),  np.sin(theta),  0],
                      [-np.sin(theta),  np.cos(theta),  0],
                      [             0,              0,  1]])
        return R

    def robot_link(self, pose, return_vertices=False):
        xPose = pose[0, 0]
        yPose = pose[1, 0]
        tPose = pose[2, 0]

        # body
        v1 = np.array([-self.d, -self.l]).reshape(2, 1)
        v2 = np.array([ self.d, -self.l]).reshape(2, 1)
        v3 = np.array([ self.d,  self.l]).reshape(2, 1)
        v4 = np.array([-self.d,  self.l]).reshape(2, 1)

        # wheel
        w1 = np.array([-self.d-self.r, -self.l]).reshape(2, 1)
        w2 = np.array([ self.d+self.r, -self.l]).reshape(2, 1)
        w3 = np.array([ self.d+self.r,  self.l]).reshape(2, 1)
        w4 = np.array([-self.d-self.r,  self.l]).reshape(2, 1)

        v1 = rbt.rot2d(tPose) @ v1 + np.array([xPose, yPose]).reshape(2, 1)
        v2 = rbt.rot2d(tPose) @ v2 + np.array([xPose, yPose]).reshape(2, 1)
        v3 = rbt.rot2d(tPose) @ v3 + np.array([xPose, yPose]).reshape(2, 1)
        v4 = rbt.rot2d(tPose) @ v4 + np.array([xPose, yPose]).reshape(2, 1)

        w1 = rbt.rot2d(tPose) @ w1 + np.array([xPose, yPose]).reshape(2, 1)
        w2 = rbt.rot2d(tPose) @ w2 + np.array([xPose, yPose]).reshape(2, 1)
        w3 = rbt.rot2d(tPose) @ w3 + np.array([xPose, yPose]).reshape(2, 1)
        w4 = rbt.rot2d(tPose) @ w4 + np.array([xPose, yPose]).reshape(2, 1)

        if return_vertices:
            return v1, v2, v3, v4, w1, w2, w3, w4
        else:
            return [[  xPose,   yPose],
                    [v1[0,0], v1[1,0]],
                    [v2[0,0], v2[1,0]],
                    [v3[0,0], v3[1,0]],
                    [v4[0,0], v4[1,0]],
                    [w1[0,0], w1[1,0]],
                    [w2[0,0], w2[1,0]],
                    [w3[0,0], w3[1,0]],
                    [w4[0,0], w4[1,0]]]

    def plot_robot(self, pose, axis):
        v1, v2, v3, v4, w1, w2, w3, w4 = self.robot_link(pose, return_vertices=True)

        axis.plot([v1[0, 0], v2[0, 0]], [v1[1, 0], v2[1, 0]], c=self.color)
        axis.plot([v2[0, 0], v3[0, 0]], [v2[1, 0], v3[1, 0]], c=self.color)
        axis.plot([v3[0, 0], v4[0, 0]], [v3[1, 0], v4[1, 0]], c=self.color)
        axis.plot([v4[0, 0], v1[0, 0]], [v4[1, 0], v1[1, 0]], c=self.color)

        axis.plot([pose[0, 0], v2[0, 0]], [pose[1, 0], v2[1, 0]], c=self.color)
        axis.plot([pose[0, 0], v3[0, 0]], [pose[1, 0], v3[1, 0]], c=self.color)

        axis.plot([v1[0, 0], w1[0, 0]], [v1[1, 0], w1[1, 0]], c='red')
        axis.plot([v2[0, 0], w2[0, 0]], [v2[1, 0], w1[1, 0]], c='red')
        axis.plot([v3[0, 0], w3[0, 0]], [v3[1, 0], w3[1, 0]], c='red')
        axis.plot([v4[0, 0], w4[0, 0]], [v4[1, 0], w4[1, 0]], c='red')

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from matplotlib import animation

    robot = MecanumDrive(wheelRadius=0.2, baseLength=0.5, baseWidth=0.5)
    pose = np.array([0, 0, 0]).reshape(3, 1)

    fig, ax = plt.subplots(1, 1)
    ax.grid(True)
    ax.set_aspect("equal")
    ax.set_xlim([-10, 10])
    ax.set_ylim([-10, 10])
    robot.plot_robot(pose, ax)
    plt.show()


    # simulation params, save history
    q = np.array([0, 0, 0]).reshape(3, 1)
    t = 0
    Ts = 0.03
    qHistCur = q.copy()
    tHist = [0]

    while t < 1000:
        # control
        bodyVeloControl = np.array([0.5, 0.5, 0.0]).reshape(3,1)

        # store history
        qHistCur = np.hstack((qHistCur, q))
        tHist.append(t * Ts)

        # Euler Intergral Update new path
        dq = robot.forward_external_kinematic(bodyVeloControl, q[2, 0])
        q = q + dq * Ts
        t += 1

    # plot
    qHistCur = qHistCur.T
    fig, ax = plt.subplots()
    ax.grid(True)
    ax.set_aspect("equal")
    ax.set_xlim([-10, 10])
    ax.set_ylim([-10, 10])
    link1, = ax.plot([], [], 'teal')
    link2, = ax.plot([], [], 'olive')
    link3, = ax.plot([], [], 'teal')
    link4, = ax.plot([], [], 'olive')
    link5, = ax.plot([], [], 'olive')
    link6, = ax.plot([], [], 'teal')
    wheel1, = ax.plot([], [], 'black')
    wheel2, = ax.plot([], [], 'black')
    wheel3, = ax.plot([], [], 'black')
    wheel4, = ax.plot([], [], 'black')

    def update(frame):
        link = robot.robot_link(qHistCur[frame].reshape(3, 1))
        link1.set_data([link[0][0], link[2][0]], [link[0][1], link[2][1]])
        link2.set_data([link[1][0], link[2][0]], [link[1][1], link[2][1]])
        link3.set_data([link[2][0], link[3][0]], [link[2][1], link[3][1]])
        link4.set_data([link[3][0], link[4][0]], [link[3][1], link[4][1]])
        link5.set_data([link[4][0], link[1][0]], [link[4][1], link[1][1]])
        link6.set_data([link[0][0], link[3][0]], [link[0][1], link[3][1]])

        wheel1.set_data([link[5][0], link[1][0]], [link[5][1], link[1][1]])
        wheel2.set_data([link[6][0], link[2][0]], [link[6][1], link[2][1]])
        wheel3.set_data([link[7][0], link[3][0]], [link[7][1], link[3][1]])
        wheel4.set_data([link[8][0], link[4][0]], [link[8][1], link[4][1]])

    animation = animation.FuncAnimation(fig, update, frames=(qHistCur.shape[0]), interval=10)
    plt.show()