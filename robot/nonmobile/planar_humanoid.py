import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.transforms import Affine2D
from matplotlib.animation import FuncAnimation
import matplotlib.patches as mpatches
from scipy.optimize import least_squares


class PlanarHumanoid:

    def __init__(self, l1, l2, l3, l4, l5, l6):
        self.l1 = l1
        self.l2 = l2
        self.l3 = l3
        self.l4 = l4
        self.l5 = l5
        self.l6 = l6

        self.ml = 0.4
        self.mw = 0.1

    def forward_kinematic_right(self, q):
        y = q[0]
        t = q[1]
        th1 = q[2]
        th2 = q[3]
        th3 = q[4]
        th4 = q[5]
        th5 = q[6]
        th6 = q[7]

        xee = (
            self.l1 * np.cos(t + th1)
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

    def forward_kinematic_left(self, q):
        y = q[0]
        t = q[1]
        th1 = q[2]
        th2 = q[3]
        th3 = q[4]
        th4 = q[5]
        th5 = q[6]
        th6 = q[7]

        xee = (
            self.l1 * np.cos(t + th1)
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

    def forward_kinematic_both(self, q):
        xee_r = self.forward_kinematic_right(q[:8])
        xee_l = self.forward_kinematic_left(q[8:])
        return np.concatenate([xee_r, xee_l])

    def inverse_kinematic_right(self):
        pass

    def inverse_kinematic_left(self):
        pass

    def inverse_kinematic_both(self):
        pass

    def jacobian(self):
        pass

    def _forward_kinematic_links(self, q):
        """
        !DO NOT MODIFY THIS METHOD
        """
        x = 0
        y = q[0]
        t = q[1]
        th1 = q[2]
        th2 = q[3]
        th3 = q[4]
        th4 = q[5]
        th5 = q[6]
        th6 = q[7]
        th1p = q[8]
        th2p = q[9]
        th3p = q[10]
        th4p = q[11]
        th5p = q[12]
        th6p = q[13]

        # base
        x0 = x
        y0 = y

        # right arm
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

        # left arm
        x1p = x0 + self.l1 * np.cos(t + th1p)
        y1p = y0 + self.l1 * np.sin(t + th1p)
        x2p = x1p + self.l2 * np.cos(t + th1p + th2p)
        y2p = y1p + self.l2 * np.sin(t + th1p + th2p)
        x3p = x2p + self.l3 * np.cos(t + th1p + th2p + th3p)
        y3p = y2p + self.l3 * np.sin(t + th1p + th2p + th3p)
        x4p = x3p + self.l4 * np.cos(t + th1p + th2p + th3p + th4p)
        y4p = y3p + self.l4 * np.sin(t + th1p + th2p + th3p + th4p)
        x5p = x4p + self.l5 * np.cos(t + th1p + th2p + th3p + th4p + th5p)
        y5p = y4p + self.l5 * np.sin(t + th1p + th2p + th3p + th4p + th5p)
        x6p = x5p + self.l6 * np.cos(t + th1p + th2p + th3p + th4p + th5p + th6p)
        y6p = y5p + self.l6 * np.sin(t + th1p + th2p + th3p + th4p + th5p + th6p)

        return (
            (x0, y0),
            (x1, y1),
            (x2, y2),
            (x3, y3),
            (x4, y4),
            (x5, y5),
            (x6, y6),
            (x1p, y1p),
            (x2p, y2p),
            (x3p, y3p),
            (x4p, y4p),
            (x5p, y5p),
            (x6p, y6p),
        )

    def plot_robot(self, q):
        fig, ax = plt.subplots()
        ax.set_aspect("equal")
        ax.grid(True)
        ax.set_xlim(-2, 2)
        ax.set_ylim(-2, 2)

        (
            (x0, y0),
            (x1, y1),
            (x2, y2),
            (x3, y3),
            (x4, y4),
            (x5, y5),
            (x6, y6),
            (x1p, y1p),
            (x2p, y2p),
            (x3p, y3p),
            (x4p, y4p),
            (x5p, y5p),
            (x6p, y6p),
        ) = self._forward_kinematic_links(q)

        # base
        center = (x0, y0)
        rotation_angle = np.degrees(q[1])
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

        # arm left
        ax.plot(
            [x0, x1, x2, x3, x4, x5, x6],
            [y0, y1, y2, y3, y4, y5, y6],
            "-o",
            color="red",
            lw=2,
            markersize=4,
        )

        # arm right
        ax.plot(
            [x0, x1p, x2p, x3p, x4p, x5p, x6p],
            [y0, y1p, y2p, y3p, y4p, y5p, y6p],
            "-o",
            color="blue",
            lw=2,
            markersize=4,
        )

        plt.show()


if __name__ == "__main__":
    robot = PlanarHumanoid(0.3, 0.3, 0.3, 0.3, 0.2, 0.1)
    q = np.array(
        [
            0.3,
            np.pi / 4,
            np.pi / 6,
            -np.pi / 6,
            np.pi / 6,
            -np.pi / 6,
            np.pi / 6,
            -np.pi / 6,
            np.pi / 6,
            -np.pi / 6,
            np.pi / 6,
            -np.pi / 6,
            np.pi / 6,
            -np.pi / 6,
        ]
    )
    q[8] = q[8] + np.pi
    robot.plot_robot(q)
