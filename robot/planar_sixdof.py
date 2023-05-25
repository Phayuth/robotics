import numpy as np


class PlanarSixDof:

    def __init__(self):
        self.a1 = 0.5
        self.a2 = 0.5
        self.a3 = 0.5
        self.a4 = 0.5
        self.a5 = 0.5
        self.a6 = 0.5

    def forward_kinematic(self, theta, return_link_pos=False):
        theta1 = theta[0, 0]
        theta2 = theta[1, 0]
        theta3 = theta[2, 0]
        theta4 = theta[3, 0]
        theta5 = theta[4, 0]
        theta6 = theta[5, 0]

        if return_link_pos:
            link_end_pose = []

            x0 = 0
            y0 = 0
            link_end_pose.append([x0, y0])

            # link 1 pose
            x1 = x0 + self.a1 * np.cos(theta1)
            y1 = y0 + self.a1 * np.sin(theta1)
            link_end_pose.append([x1, y1])

            # link 2 pose
            x2 = x1 + self.a2 * np.cos(theta1 + theta2)
            y2 = y1 + self.a2 * np.sin(theta1 + theta2)
            link_end_pose.append([x2, y2])

            # link 3 pose
            x3 = x2 + self.a3 * np.cos(theta1 + theta2 + theta3)
            y3 = y2 + self.a3 * np.sin(theta1 + theta2 + theta3)
            link_end_pose.append([x3, y3])

            # link 4 pose
            x4 = x3 + self.a4 * np.cos(theta1 + theta2 + theta3 + theta4)
            y4 = y3 + self.a4 * np.sin(theta1 + theta2 + theta3 + theta4)
            link_end_pose.append([x4, y4])

            # link 5 pose
            x5 = x4 + self.a5 * np.cos(theta1 + theta2 + theta3 + theta4 + theta5)
            y5 = y4 + self.a5 * np.sin(theta1 + theta2 + theta3 + theta4 + theta5)
            link_end_pose.append([x5, y5])

            # link 6 pose
            x6 = x5 + self.a6 * np.cos(theta1 + theta2 + theta3 + theta4 + theta5 + theta6)
            y6 = y5 + self.a6 * np.sin(theta1 + theta2 + theta3 + theta4 + theta5 + theta6)
            link_end_pose.append([x6, y6])

            return link_end_pose

        else:

            x = self.a1 * np.cos(theta1) + self.a2 * np.cos(theta1 + theta2) + self.a3 * np.cos(theta1 + theta2 + theta3) + self.a4 * np.cos(theta1 + theta2 + theta3 + theta4) + self.a5 * np.cos(theta1 + theta2 + theta3 + theta4 + theta5) + self.a6 * np.cos(theta1 + theta2 + theta3 + theta4 + theta5 + theta6)
            y = self.a1 * np.sin(theta1) + self.a2 * np.sin(theta1 + theta2) + self.a3 * np.sin(theta1 + theta2 + theta3) + self.a4 * np.sin(theta1 + theta2 + theta3 + theta4) + self.a5 * np.sin(theta1 + theta2 + theta3 + theta4 + theta5) + self.a6 * np.sin(theta1 + theta2 + theta3 + theta4 + theta5 + theta6)
            
            return np.array([[x], [y]])

    def plot_arm(self, theta, plt_axis=None):
        theta1 = theta[0, 0]
        theta2 = theta[1, 0]
        theta3 = theta[2, 0]
        theta4 = theta[3, 0]
        theta5 = theta[4, 0]
        theta6 = theta[5, 0]

        x0 = 0
        y0 = 0

        # link 1 pose
        x1 = x0 + self.a1 * np.cos(theta1)
        y1 = y0 + self.a1 * np.sin(theta1)

        # link 2 pose
        x2 = x1 + self.a2 * np.cos(theta1 + theta2)
        y2 = y1 + self.a2 * np.sin(theta1 + theta2)

        # link 3 pose
        x3 = x2 + self.a3 * np.cos(theta1 + theta2 + theta3)
        y3 = y2 + self.a3 * np.sin(theta1 + theta2 + theta3)

        # link 4 pose
        x4 = x3 + self.a4 * np.cos(theta1 + theta2 + theta3 + theta4)
        y4 = y3 + self.a4 * np.sin(theta1 + theta2 + theta3 + theta4)

        # link 5 pose
        x5 = x4 + self.a5 * np.cos(theta1 + theta2 + theta3 + theta4 + theta5)
        y5 = y4 + self.a5 * np.sin(theta1 + theta2 + theta3 + theta4 + theta5)

        # link 6 pose
        x6 = x5 + self.a6 * np.cos(theta1 + theta2 + theta3 + theta4 + theta5 + theta6)
        y6 = y5 + self.a6 * np.sin(theta1 + theta2 + theta3 + theta4 + theta5 + theta6)
        
        if plt_axis:
            plt_axis.axvline(x=0, c="green")
            plt_axis.axhline(y=0, c="green")

            plt_axis.plot([x0, x1], [y0, y1], 'cyan', linewidth=3)
            plt_axis.plot([x1, x2], [y1, y2], 'tan', linewidth=3)
            plt_axis.plot([x2, x3], [y2, y3], 'olive', linewidth=3)
            plt_axis.plot([x3, x4], [y3, y4], 'navy', linewidth=3)
            plt_axis.plot([x4, x5], [y4, y5], 'lime', linewidth=3)
            plt_axis.plot([x5, x6], [y5, y6], 'peru', linewidth=3)

            plt_axis.plot([x6], [y6], 'ro')


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    robot = PlanarSixDof()
    fig, ax = plt.subplots()
    ax.set_aspect("equal")
    t1 = np.linspace(0,np.pi/2,100)
    for i in range(t1.shape[0]):
        robot.plot_arm(np.array([i,0,0,0,0,0]).reshape(6, 1), plt_axis=ax)
        plt.pause(1)
    plt.show()
    