import numpy as np
import matplotlib.pyplot as plt


class PlanarRR:

    def __init__(self):
        self.alpha1 = 0
        self.alpha2 = 0
        self.d1 = 0
        self.d2 = 0
        self.a1 = 2
        self.a2 = 2

    def dh_transformation(theta,alpha,d,a):
        R = np.array([[np.cos(theta), -np.sin(theta)*np.cos(alpha),  np.sin(theta)*np.sin(alpha), a*np.cos(theta)],
                      [np.sin(theta),  np.cos(theta)*np.cos(alpha), -np.cos(theta)*np.sin(alpha), a*np.sin(theta)],
                      [            0,                np.sin(alpha),                np.cos(alpha),               d],
                      [            0,                            0,                            0,               1]])
        return R

    def forward_kinematic(self, theta, return_link_pos=False):
        theta1 = theta[0, 0]
        theta2 = theta[1, 0]

        x = self.a1 * np.cos(theta1) + self.a2 * np.cos(theta1 + theta2)
        y = self.a1 * np.sin(theta1) + self.a2 * np.sin(theta1 + theta2)

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

            return link_end_pose

        else:
            return np.array([[x], [y]])

    def inverse_kinematic_geometry(self, desired_pose, elbow_option):
        x = desired_pose[0, 0]
        y = desired_pose[1, 0]

        # check if the desired pose is inside of task space or not
        rd = np.sqrt(x**2 + y**2)
        link_length = self.a1 + self.a2
        if rd > link_length:
            print("The desired pose is outside of taskspace")
            return None

        if elbow_option == 0:
            sign = -1
        elif elbow_option == 1:
            sign = 1

        D = (x**2 + y**2 - self.a1**2 - self.a2**2) / (2 * self.a1 * self.a2)

        theta2 = np.arctan2(sign * np.sqrt(1 - D**2), D)

        theta1 = np.arctan2(y, x) - np.arctan2((self.a2 * np.sin(theta2)), (self.a1 + self.a2 * np.cos(theta2)))

        return np.array([[theta1], [theta2]])

    def jacobian(self, theta):
        theta1 = theta[0, 0]
        theta2 = theta[1, 0]

        J = np.array([[-self.a1 * np.sin(theta1) - self.a2 * np.sin(theta1 + theta2), -self.a2 * np.sin(theta1 + theta2)],
                      [ self.a1 * np.cos(theta1) + self.a2 * np.cos(theta1 + theta2),  self.a2 * np.cos(theta1 + theta2)]])
        return J

    def forward_kinematic_dh(self, theta):
        theta1 = theta[0, 0]
        theta2 = theta[1, 0]

        A1 = self.dh_transformation(theta1, self.alpha1, self.d1, self.a1)
        A2 = self.dh_transformation(theta2, self.alpha2, self.d2, self.a2)

        T02 = A1 @ A2

        return T02

    def jacobian_dh(self, theta):
        theta1 = theta[0, 0]
        theta2 = theta[1, 0]

        O0 = np.array([[0], [0], [0]])
        O1 = np.array([[self.a1 * np.cos(theta1)], [self.a1 * np.sin(theta1)], [0]])
        O2 = np.array([[self.a1 * np.cos(theta1) + self.a2 * np.cos(theta1 + theta2)], [self.a1 * np.sin(theta1) + self.a2 * np.sin(theta1 + theta2)], [0]])
        Z0 = np.array([[0], [0], [1]])
        Z1 = np.array([[0], [0], [1]])

        # joint 1
        # first transpose both row vector , then do cross product , the transpose to column vector back.
        # because of np.cross use row vector (I dont know how to use in properly yet)
        Jv1 = np.transpose(np.cross(np.transpose(Z0), np.transpose(O2 - O0)))
        Jw1 = Z0

        # joint 2
        Jv2 = np.transpose(np.cross(np.transpose(Z1), np.transpose(O2 - O1)))
        Jw2 = Z1

        J1 = np.append(Jv1, Jw1, axis=0)  # if not use axis = the result is 1x6, axis=0 the result is 6x1, axis=1 the result is 3x2
        J2 = np.append(Jv2, Jw2, axis=0)
        J = np.append(J1, J2, axis=1)  # we want J = [J1 , J2] side by side => use axis = 1

        return J

    def plot_arm(self, theta, plt_basis=False, plt_show=False):
        theta1 = theta[0, 0]
        theta2 = theta[1, 0]

        shoulder = np.array([0, 0])
        elbow = shoulder + np.array([self.a1 * np.cos(theta1), self.a1 * np.sin(theta1)])
        wrist = elbow + np.array([self.a2 * np.cos(theta1 + theta2), self.a2 * np.sin(theta1 + theta2)])

        if plt_basis:
            plt.axes().set_aspect('equal')
            plt.axvline(x=0, c="green")
            plt.axhline(y=0, c="green")

        plt.plot([shoulder[0], elbow[0]], [shoulder[1], elbow[1]], color='indigo', linewidth=5, marker='o', markerfacecolor='r')
        plt.plot([elbow[0], wrist[0]], [elbow[1], wrist[1]], color='indigo', linewidth=5, marker='o', markerfacecolor='r')

        # plt.plot(shoulder[0], shoulder[1], 'ro')
        # plt.plot(elbow[0], elbow[1], 'ro')
        # plt.plot(wrist[0], wrist[1], 'ro')

        if plt_show:
            plt.show()


class PlanarRRVoxel(object):

    def __init__(self, base_position=None, link_lenths=None):

        self.base_position = base_position
        self.link_lenths = np.array(link_lenths)

    def robot_position(self, theta1, theta2):

        position = []
        position.append(self.base_position)

        theta1 = (theta1 * np.pi) / 180
        theta2 = (theta2 * np.pi) / 180

        x1 = self.base_position[0] + self.link_lenths[0] * np.cos(theta1)
        y1 = self.base_position[1] + self.link_lenths[0] * np.sin(theta1)

        position.append([x1, y1])

        x2 = self.base_position[0] + self.link_lenths[0] * np.cos(theta1) + self.link_lenths[1] * np.cos(theta1 + theta2)
        y2 = self.base_position[1] + self.link_lenths[0] * np.sin(theta1) + self.link_lenths[1] * np.sin(theta1 + theta2)

        position.append([x2, y2])

        return position


if __name__ == "__main__":
    base_position = [15, 15]
    link_lenths = [5, 5]
    robot = PlanarRRVoxel(base_position, link_lenths)

    plt.figure(figsize=(10, 10))
    plt.axes().set_aspect('equal')
    r1 = robot.robot_position(90, 0)
    plt.plot([robot.base_position[0], r1[0][0]], [robot.base_position[1], r1[0][1]], "b", linewidth=8)
    plt.plot([r1[0][0], r1[1][0]], [r1[0][1], r1[1][1]], "b", linewidth=8)
    plt.plot([r1[1][0], r1[2][0]], [r1[1][1], r1[2][1]], "r", linewidth=8)
    plt.gca().invert_yaxis()
    plt.show()