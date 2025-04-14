import numpy as np


class PlanarRRR:

    def __init__(self):
        # kinematic
        # self.a1 = 1
        # self.a2 = 1
        # self.a3 = 0.5
        self.a1 = 1.6
        self.a2 = 1.6
        self.a3 = 0.8

        # gripper
        self.gripperLength = 0.2
        self.gripperWidth = 0.2
        self.gripperOffset = 0.1

        # visual
        self.shadowlevel = 0.2

        self.basecolor = "black"
        self.baselinewidth = 1
        self.basemarker = "s"
        self.basemarkerfacecolor = "white"

        self.linkcolor = "indigo"
        self.linkwidth = 1

        self.jointmarker = "o"
        self.jointcolor = "k"
        self.jointlinewidth = 0
        self.jointmarkersize = 2
        self.jointmarkerfacecolor = "white"

        self.eemarker = "o"
        self.eecolor = "white"
        self.eelinewidth = 1
        self.eemarkersize = 2
        self.eemarkerfacecolor = "indigo"

    def forward_kinematic(self, theta, return_link_pos=False):
        theta1 = theta[0, 0]
        theta2 = theta[1, 0]
        theta3 = theta[2, 0]

        H01 = np.array([[np.cos(theta1), -np.sin(theta1), 0, 0], [np.sin(theta1), np.cos(theta1), 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        H12 = np.array([[np.cos(theta2), -np.sin(theta2), 0, self.a1], [np.sin(theta2), np.cos(theta2), 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        H23 = np.array([[np.cos(theta3), -np.sin(theta3), 0, self.a2], [np.sin(theta3), np.cos(theta3), 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        H34 = np.array([[1, 0, 0, self.a3], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        H04 = H01 @ H12 @ H23 @ H34
        phi = theta1 + theta2 + theta3

        if return_link_pos:

            # option for return link end pose. normally used for collision checking
            link_end_pose = []
            link_end_pose.append([0, 0])

            # link 1 pose
            x1 = self.a1 * np.cos(theta1)
            y1 = self.a1 * np.sin(theta1)
            link_end_pose.append([x1, y1])

            # link 2 pose
            x2 = self.a1 * np.cos(theta1) + self.a2 * np.cos(theta1 + theta2)
            y2 = self.a1 * np.sin(theta1) + self.a2 * np.sin(theta1 + theta2)
            link_end_pose.append([x2, y2])

            # link 3 pose
            x3 = self.a1 * np.cos(theta1) + self.a2 * np.cos(theta1 + theta2) + self.a3 * np.cos(theta1 + theta2 + theta3)
            y3 = self.a1 * np.sin(theta1) + self.a2 * np.sin(theta1 + theta2) + self.a3 * np.sin(theta1 + theta2 + theta3)
            link_end_pose.append([x3, y3])

            return link_end_pose

        else:
            return np.array([[H04[0, 3]], [H04[1, 3]], [phi]])

    def foward_kinematic_with_gripper(self, theta, return_link_pos=False):
        theta1 = theta[0, 0]
        theta2 = theta[1, 0]
        theta3 = theta[2, 0]

        x = self.a1 * np.cos(theta1) + self.a2 * np.cos(theta1 + theta2) + (self.a3 + self.gripperLength - self.gripperOffset) * np.cos(theta1 + theta2 + theta3)
        y = self.a1 * np.sin(theta1) + self.a2 * np.sin(theta1 + theta2) + (self.a3 + self.gripperLength - self.gripperOffset) * np.sin(theta1 + theta2 + theta3)
        phi = theta1 + theta2 + theta3

        if return_link_pos:
            # base
            link_end_pose = []
            link_end_pose.append([0, 0])

            # link 1 pose
            x1 = self.a1 * np.cos(theta1)
            y1 = self.a1 * np.sin(theta1)
            link_end_pose.append([x1, y1])

            # link 2 pose
            x2 = x1 + self.a2 * np.cos(theta1 + theta2)
            y2 = y1 + self.a2 * np.sin(theta1 + theta2)
            link_end_pose.append([x2, y2])

            # link 3 pose
            x3 = x2 + self.a3 * np.cos(theta1 + theta2 + theta3)
            y3 = y2 + self.a3 * np.sin(theta1 + theta2 + theta3)
            link_end_pose.append([x3, y3])

            # end tip
            x4 = x3 + self.gripperLength * np.cos(theta1 + theta2 + theta3)
            y4 = y3 + self.gripperLength * np.sin(theta1 + theta2 + theta3)
            link_end_pose.append([x4, y4])

            # tcp
            xtcp = x3 + (self.gripperLength - self.gripperOffset) * np.cos(theta1 + theta2 + theta3)
            ytcp = y3 + (self.gripperLength - self.gripperOffset) * np.sin(theta1 + theta2 + theta3)
            link_end_pose.append([xtcp, ytcp])

            # base left jaw
            xblj = x3 + (self.gripperWidth / 2) * np.cos(theta1 + theta2 + theta3 + np.pi / 2)
            yblj = y3 + (self.gripperWidth / 2) * np.sin(theta1 + theta2 + theta3 + np.pi / 2)
            link_end_pose.append([xblj, yblj])

            # base right jaw
            xbrj = x3 + (self.gripperWidth / 2) * np.cos(theta1 + theta2 + theta3 - np.pi / 2)
            ybrj = y3 + (self.gripperWidth / 2) * np.sin(theta1 + theta2 + theta3 - np.pi / 2)
            link_end_pose.append([xbrj, ybrj])

            # tip left jaw
            xtlj = xblj + self.gripperLength * np.cos(theta1 + theta2 + theta3)
            ytlj = yblj + self.gripperLength * np.sin(theta1 + theta2 + theta3)
            link_end_pose.append([xtlj, ytlj])

            # tip right jaw
            xtrj = xbrj + self.gripperLength * np.cos(theta1 + theta2 + theta3)
            ytrj = ybrj + self.gripperLength * np.sin(theta1 + theta2 + theta3)
            link_end_pose.append([xtrj, ytrj])

            # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
            # [base, link1, link2, link3, endtip, tcp, baseleftjaw, baserightjaw, tipleftjaw, tiprightjaw]
            return link_end_pose
        else:
            return np.array([[x], [y], [phi]])

    def inverse_kinematic_geometry(self, desiredConfig, elbow_option=0, with_gripper=False):
        x = desiredConfig[0, 0]
        y = desiredConfig[1, 0]
        phi = desiredConfig[2, 0]

        # modify the length of the third link if gripper is used
        length3 = self.a3 + self.gripperLength - self.gripperOffset if with_gripper else self.a3

        x2 = x - length3 * np.cos(phi)
        y2 = y - length3 * np.sin(phi)

        t2term = (x2**2 + y2**2 - self.a1**2 - self.a2**2) / (2 * self.a1 * self.a2)
        t2term = np.clip(t2term, -1, 1)  # Ensure it's within the valid domain

        if elbow_option == 0:
            sign = -1
        elif elbow_option == 1:
            sign = 1

        theta2 = sign * np.arccos(t2term)  # positive for elbow down, negative for elbow up
        theta1 = np.arctan2(y2, x2) - np.arctan2(self.a2 * np.sin(theta2), self.a1 + self.a2 * np.cos(theta2))
        theta3 = phi - theta1 - theta2

        return np.array([[theta1], [theta2], [theta3]])

    def jacobian(self, theta):
        theta1 = theta[0, 0]
        theta2 = theta[1, 0]
        theta3 = theta[2, 0]

        J = np.array(
            [
                [-self.a1 * np.sin(theta1) - self.a2 * np.sin(theta1 + theta2) - self.a3 * np.sin(theta1 + theta2 + theta3), -self.a2 * np.sin(theta1 + theta2) - self.a3 * np.sin(theta1 + theta2 + theta3), -self.a3 * np.sin(theta1 + theta2 + theta3)],
                [self.a1 * np.cos(theta3) + self.a2 * np.cos(theta1 + theta2) + self.a3 * np.cos(theta1 + theta2 + theta3), self.a2 * np.cos(theta1 + theta2) + self.a3 * np.cos(theta1 + theta2 + theta3), self.a3 * np.cos(theta1 + theta2 + theta3)],
                [1, 1, 1],
            ]
        )

        return J

    def _plot_single(self, theta, axis, color, shadow=False):
        links = self.forward_kinematic(theta, return_link_pos=True)

        # link color
        axis.plot(
            [links[0][0], links[1][0], links[2][0], links[3][0]],
            [links[0][1], links[1][1], links[2][1], links[3][1]],
            color=color,
            linewidth=self.linkwidth,
            alpha=self.shadowlevel if shadow else 1.0,
        )

        # joint color
        axis.plot(
            [links[1][0], links[2][0]],
            [links[1][1], links[2][1]],
            color=self.jointcolor,
            linewidth=self.jointlinewidth,
            marker=self.jointmarker,
            markerfacecolor=self.jointmarkerfacecolor,
            markersize=self.jointmarkersize,
            alpha=self.shadowlevel if shadow else 1.0,
        )

        # end effector color
        axis.plot(
            [links[3][0]],
            [links[3][1]],
            color=self.eecolor,
            linewidth=self.eelinewidth,
            marker=self.eemarker,
            markerfacecolor=self.eemarkerfacecolor,
            markersize=self.eemarkersize,
            alpha=self.shadowlevel if shadow else 1.0,
        )

    def _plot_single_with_gripper(self, theta, axis, color, shadow=False):
        link = self.foward_kinematic_with_gripper(theta, return_link_pos=True)

        # link color
        axis.plot(
            [link[0][0], link[1][0], link[2][0], link[3][0]],
            [link[0][1], link[1][1], link[2][1], link[3][1]],
            color=color,
            linewidth=self.linkwidth,
            alpha=self.shadowlevel if shadow else 1.0,
        )

        #  gripper color
        axis.plot(
            [link[8][0], link[6][0], link[7][0], link[9][0]],
            [link[8][1], link[6][1], link[7][1], link[9][1]],
            color=color,
            linewidth=self.linkwidth,
            alpha=self.shadowlevel if shadow else 1.0,
        )

        # elbow and ee marker
        axis.plot(
            [link[1][0], link[2][0], link[3][0]],
            [link[1][1], link[2][1], link[3][1]],
            color=self.jointcolor,
            linewidth=0,
            marker=self.jointmarker,
            markerfacecolor=self.jointmarkerfacecolor,
            markersize=self.jointmarkersize,
            alpha=self.shadowlevel if shadow else 1.0,
        )

        # tcp marker
        axis.plot(
            [link[5][0]],
            [link[5][1]],
            color=self.jointcolor,
            linewidth=self.jointlinewidth,
            marker=self.jointmarker,
            markerfacecolor=self.jointmarkerfacecolor,
            markersize=self.jointmarkersize,
            alpha=self.shadowlevel if shadow else 1.0,
        )

        # tip marker
        axis.plot(
            [link[4][0]],
            [link[4][1]],
            color=self.eecolor,
            linewidth=self.eelinewidth,
            marker=self.eemarker,
            markerfacecolor=self.eemarkerfacecolor,
            markersize=self.eemarkersize,
            alpha=self.shadowlevel if shadow else 1.0,
        )

    def plot_arm(self, thetas, axis, shadow=False, colors=[]):
        if thetas.shape == (3, 1):
            self._plot_single(thetas, axis, color=self.linkcolor, shadow=shadow)
        else:
            colors = colors if len(colors) != 0 else [self.linkcolor] * thetas.shape[1]
            for i in range(thetas.shape[1]):
                if i in [0, thetas.shape[1] - 1]:
                    self._plot_single(thetas[:, i, np.newaxis], axis, color=colors[i], shadow=False)
                else:
                    self._plot_single(thetas[:, i, np.newaxis], axis, color=colors[i], shadow=shadow)

        # base color
        axis.plot(
            [0],
            [0],
            color=self.basecolor,
            linewidth=self.baselinewidth,
            marker=self.basemarker,
            markerfacecolor=self.basemarkerfacecolor,
        )


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    robot = PlanarRRR()
    desiredPose = np.array([[2], [2], [0.9]])
    thetaUp = robot.inverse_kinematic_geometry(desiredPose, elbow_option=0)
    thetaDown = robot.inverse_kinematic_geometry(desiredPose, elbow_option=1)

    fig, ax = plt.subplots()
    robot.plot_arm(thetaDown, ax)
    robot._plot_single_with_gripper(thetaUp, ax, color="red", shadow=False)
    ax.set_xlim([-3, 3])
    ax.set_ylim([-3, 3])
    ax.set_aspect("equal", adjustable="box")
    plt.show()
