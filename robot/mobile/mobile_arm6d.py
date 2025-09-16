import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.transforms import Affine2D
from matplotlib.animation import FuncAnimation
import matplotlib.patches as mpatches
from scipy.optimize import least_squares


class MobileManipulator6d:

    def __init__(self, l1, l2, l3, l4, l5, l6):
        self.l1 = l1
        self.l2 = l2
        self.l3 = l3
        self.l4 = l4
        self.l5 = l5
        self.l6 = l6

        self.ml = 0.4
        self.mw = 0.3

    def forward_kinematic(self, q):
        x = q[0]
        y = q[1]
        t = q[2]
        th1 = q[3]
        th2 = q[4]
        th3 = q[5]
        th4 = q[6]
        th5 = q[7]
        th6 = q[8]

        xee = (
            x
            + self.l1 * np.cos(t + th1)
            + self.l2 * np.cos(t + th1 + th2)
            + self.l3 * np.cos(t + th1 + th2 + th3)
            + self.l4 * np.cos(t + th1 + th2 + th3 + th4)
            + self.l5 * np.cos(t + th1 + th2 + th3 + th4 + th5)
            + self.l6 * np.cos(t + th1 + th2 + th3 + th4 + th5 + th6)
        )
        yee = (
            y
            + self.l1 * np.sin(t + th1)
            + self.l2 * np.sin(t + th1 + th2)
            + self.l3 * np.sin(t + th1 + th2 + th3)
            + self.l4 * np.sin(t + th1 + th2 + th3 + th4)
            + self.l5 * np.sin(t + th1 + th2 + th3 + th4 + th5)
            + self.l6 * np.sin(t + th1 + th2 + th3 + th4 + th5 + th6)
        )
        tee = t + th1 + th2 + th3 + th4 + th5 + th6
        return np.array([xee, yee, tee])

    def jacobian(self, q):
        pass

    def inverse_kinematic(self, target, q0):

        def residual(q, target):
            return self.forward_kinematic(q) - target

        result = least_squares(residual, q0, args=(target,), method="trf")
        return result.success, result.x

    def _forward_kinematic_links(self, q):
        """
        !DO NOT MODIFY THIS METHOD
        """
        x = q[0]
        y = q[1]
        t = q[2]
        th1 = q[3]
        th2 = q[4]
        th3 = q[5]
        th4 = q[6]
        th5 = q[7]
        th6 = q[8]

        x0 = x
        y0 = y
        x1 = x0 + self.l1 * np.cos(t + th1)
        y1 = y0 + self.l1 * np.sin(t + th1)
        x2 = x1 + self.l2 * np.cos(t + th1 + th2)
        y2 = y1 + self.l2 * np.sin(t + th1 + th2)
        x3 = x2 + self.l3 * np.cos(t + th1 + th2 + th3)
        y3 = y2 + self.l3 * np.sin(t + th1 + th2 + th3)
        x4 = x3 + self.l4 * np.cos(t + th1 + th2 + th3 + th4)
        y4 = y3 + self.l4 * np.sin(t + th1 + th2 + th3 + th4)
        x5 = x4 + self.l5 * np.cos(t + th1 + th2 + th3 + th4 + th5)
        y5 = y4 + self.l5 * np.sin(t + th1 + th2 + th3 + th4 + th5)
        x6 = x5 + self.l6 * np.cos(t + th1 + th2 + th3 + th4 + th5 + th6)
        y6 = y5 + self.l6 * np.sin(t + th1 + th2 + th3 + th4 + th5 + th6)
        return (x0, y0), (x1, y1), (x2, y2), (x3, y3), (x4, y4), (x5, y5), (x6, y6)

    def plot_robot(self, q):
        fig, ax = plt.subplots()
        ax.set_aspect("equal")
        ax.grid(True)
        ax.set_xlim(-2, 2)
        ax.set_ylim(-2, 2)

        x = q[0]
        y = q[1]
        t = q[2]
        (x0, y0), (x1, y1), (x2, y2), (x3, y3), (x4, y4), (x5, y5), (x6, y6) = (
            self._forward_kinematic_links(q)
        )

        # base
        center = (x, y)
        rotation_angle = np.degrees(t)
        rectangle = patches.Rectangle(
            (center[0] - self.ml / 2, center[1] - self.mw / 2),
            self.ml,
            self.mw,
            edgecolor="black",
            facecolor="lightgray",
            linewidth=2,
        )
        transf = Affine2D().rotate_deg_around(center[0], center[1], rotation_angle)
        rectangle.set_transform(transf + ax.transData)
        ax.add_patch(rectangle)

        # heading arrow
        arrow_length = self.ml / 2
        dx = arrow_length * np.cos(t)
        dy = arrow_length * np.sin(t)
        arrow = mpatches.FancyArrowPatch(
            (x, y), (x + dx, y + dy), mutation_scale=20, color="blue", linewidth=2
        )
        ax.add_patch(arrow)

        # arm
        ax.plot(
            [x0, x1, x2, x3, x4, x5, x6],
            [y0, y1, y2, y3, y4, y5, y6],
            "-o",
            color="red",
            lw=2,
            markersize=4,
        )

        plt.show()

    def plot_robot_animation(self, Q):
        """
        Q: trajectory of joint space
        """
        fig, ax = plt.subplots()
        ax.set_aspect("equal")
        ax.grid(True)
        ax.set_xlim(-3, 3)
        ax.set_ylim(-3, 3)

        # plot initial position
        q = Q[0]
        x = q[0]
        y = q[1]
        t = q[2]
        (x0, y0), (x1, y1), (x2, y2), (x3, y3), (x4, y4), (x5, y5), (x6, y6) = (
            self._forward_kinematic_links(q)
        )

        # base
        center = x, y
        rotation_angle = np.rad2deg(t)  # degrees
        rectangle = patches.Rectangle(
            (center[0] - self.ml / 2, center[1] - self.mw / 2),
            self.ml,
            self.mw,
            edgecolor="blue",
            facecolor="lightblue",
            linewidth=2,
        )
        transf = Affine2D().rotate_deg_around(center[0], center[1], rotation_angle)
        rectangle.set_transform(transf + ax.transData)
        ax.add_patch(rectangle)

        # heading arrow (FancyArrowPatch for easier update)
        arrow_length = self.ml / 2
        dx = arrow_length * np.cos(t)
        dy = arrow_length * np.sin(t)
        arrow = mpatches.FancyArrowPatch(
            (x, y), (x + dx, y + dy), mutation_scale=20, color="blue", linewidth=2
        )
        ax.add_patch(arrow)

        # arm
        (linearm,) = ax.plot(
            [x0, x1, x2, x3, x4, x5, x6],
            [y0, y1, y2, y3, y4, y5, y6],
            "-o",
            color="red",
            lw=2,
            markersize=4,
        )

        def update(i):
            q = Q[i]
            x = q[0]
            y = q[1]
            t = q[2]
            (
                (x0, y0),
                (x1, y1),
                (x2, y2),
                (x3, y3),
                (x4, y4),
                (x5, y5),
                (x6, y6),
            ) = self._forward_kinematic_links(q)

            # base
            rectangle.set_xy((x - self.ml / 2, y - self.mw / 2))
            rotation_angle = np.rad2deg(t)  # degrees
            transf = Affine2D().rotate_deg_around(x, y, rotation_angle)
            rectangle.set_transform(transf + ax.transData)

            # heading arrow
            dx = arrow_length * np.cos(t)
            dy = arrow_length * np.sin(t)
            arrow.set_positions((x, y), (x + dx, y + dy))

            # arm
            linearm.set_data(
                [x0, x1, x2, x3, x4, x5, x6], [y0, y1, y2, y3, y4, y5, y6]
            )

        ani = FuncAnimation(fig, update, frames=len(Q), interval=100)
        plt.show()


if __name__ == "__main__":
    robot = MobileManipulator6d(0.3, 0.3, 0.3, 0.3, 0.2, 0.2)
    q0 = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0])
    robot.plot_robot(q0)

    qe = np.array([2, 2, 0, 0, 0, 1, 1, 1, 1])
    Q = np.linspace(q0, qe, 100)
    robot.plot_robot_animation(Q)

    target = np.array([1, 1, np.pi / 2])
    success, qsol = robot.inverse_kinematic(target, q0)
    print("IK success:", success)
    print("IK solution:", qsol)

    Q = np.linspace(q0, qsol, 100)
    robot.plot_robot_animation(Q)
