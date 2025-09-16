import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.transforms import Affine2D
from matplotlib.animation import FuncAnimation
import matplotlib.patches as mpatches
from scipy.optimize import least_squares


class MobileManipulator2d:

    def __init__(self, l1, l2):
        self.l1 = l1
        self.l2 = l2
        self.ml = 0.4  # mobile base length
        self.mw = 0.3  # mobile base width

    def forward_kinematic(self, q):
        """
        Compute forward kinematic position of end-effector of mm
        input q = [x,y,t,theta1,theta2]
        output x = [xee, yee]
        """
        x = q[0]
        y = q[1]
        t = q[2]
        theta1 = q[3]
        theta2 = q[4]

        xee = (
            x
            + self.l1 * np.cos(theta1 + t)
            + self.l2 * np.cos(theta1 + theta2 + t)
        )
        yee = (
            y
            + self.l1 * np.sin(theta1 + t)
            + self.l2 * np.sin(theta1 + theta2 + t)
        )
        return np.array([xee, yee])

    def forward_kinematic2(self, q):
        """
        Compute forward kinematic position and orientation of end-effector of mm
        input q = [x,y,t,theta1,theta2]
        output x = [xee, yee, tee]
        """
        x = q[0]
        y = q[1]
        t = q[2]
        theta1 = q[3]
        theta2 = q[4]

        xee = (
            x
            + self.l1 * np.cos(theta1 + t)
            + self.l2 * np.cos(theta1 + theta2 + t)
        )
        yee = (
            y
            + self.l1 * np.sin(theta1 + t)
            + self.l2 * np.sin(theta1 + theta2 + t)
        )
        tee = t + theta1 + theta2
        return np.array([xee, yee, tee])

    def jacobian_whole_body(self, q):
        """
        Compute the Jacobian of the whole body
        q = [x, y, t, theta1, theta2]
        qdot = [xdot, ydot, tdot, theta1dot, theta2dot]
        qdot1 = [v, w, theta1dot, theta2dot]
        """
        x = q[0]
        y = q[1]
        t = q[2]
        theta1 = q[3]
        theta2 = q[4]

        Ja = np.array(
            [
                [
                    1,
                    0,
                    -self.l1 * np.sin(t + theta1)
                    - self.l2 * np.sin(t + theta1 + theta2),
                    -self.l1 * np.sin(t + theta1)
                    - self.l2 * np.sin(t + theta1 + theta2),
                    -self.l2 * np.sin(t + theta1 + theta2),
                ],
                [
                    0,
                    1,
                    self.l1 * np.cos(t + theta1)
                    + self.l2 * np.cos(t + theta1 + theta2),
                    self.l1 * np.cos(t + theta1)
                    + self.l2 * np.cos(t + theta1 + theta2),
                    self.l2 * np.cos(t + theta1 + theta2),
                ],
            ]
        )

        Jb = np.array(
            [
                [np.cos(t), 0, 0, 0],
                [np.sin(t), 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ]
        )

        J = Ja @ Jb
        return J

    def jacobian_whole_body2(self, q):
        """
        Compute the Jacobian of the whole body
        q = [x, y, t, theta1, theta2]
        qdot = [xdot, ydot, tdot, theta1dot, theta2dot]
        qdot1 = [v, w, theta1dot, theta2dot]
        xdot = [xeed, yeed, teed]
        """
        x = q[0]
        y = q[1]
        t = q[2]
        theta1 = q[3]
        theta2 = q[4]

        Ja = np.array(
            [
                [
                    1,
                    0,
                    -self.l1 * np.sin(t + theta1)
                    - self.l2 * np.sin(t + theta1 + theta2),
                    -self.l1 * np.sin(t + theta1)
                    - self.l2 * np.sin(t + theta1 + theta2),
                    -self.l2 * np.sin(t + theta1 + theta2),
                ],
                [
                    0,
                    1,
                    self.l1 * np.cos(t + theta1)
                    + self.l2 * np.cos(t + theta1 + theta2),
                    self.l1 * np.cos(t + theta1)
                    + self.l2 * np.cos(t + theta1 + theta2),
                    self.l2 * np.cos(t + theta1 + theta2),
                ],
                [
                    0,
                    0,
                    1,
                    1,
                    1,
                ],
            ]
        )

        Jb = np.array(
            [
                [np.cos(t), 0, 0, 0],
                [np.sin(t), 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ]
        )

        J = Ja @ Jb
        return J

    def inverse_kinematic(self, target, q0):
        """
        Solve IK using TRF method for underdetermined system
        - Penalize
            - base translation (x,y)
            - base rotation (t)
            - large joint angle (theta1, theta2)
        """

        basetrancost = 1.0
        baserotcost = 1.0
        jointcost = 0.1

        def residual(q, target):
            return self.forward_kinematic(q) - target

        result = least_squares(residual, q0, args=(target,), method="trf")
        return result.success, result.x

    def inverse_kinematic2(self, target, q0):

        def residual(q, target):
            return self.forward_kinematic2(q) - target

        result = least_squares(residual, q0, args=(target,), method="trf")
        return result.success, result.x

    def _forward_kinematic_links(self, q):
        """
        !DO NOT MODIFY THIS METHOD
        """
        x = q[0]
        y = q[1]
        t = q[2]
        theta1 = q[3]
        theta2 = q[4]

        x0, y0 = x, y
        x1 = x + self.l1 * np.cos(theta1 + t)
        y1 = y + self.l1 * np.sin(theta1 + t)
        x2 = (
            x
            + self.l1 * np.cos(theta1 + t)
            + self.l2 * np.cos(theta1 + theta2 + t)
        )
        y2 = (
            y
            + self.l1 * np.sin(theta1 + t)
            + self.l2 * np.sin(theta1 + theta2 + t)
        )
        return (x0, y0), (x1, y1), (x2, y2)

    def plot_robot(self, q):
        fig, ax = plt.subplots()
        ax.set_aspect("equal")
        ax.grid(True)
        ax.set_xlim(-3, 3)
        ax.set_ylim(-3, 3)

        x = q[0]
        y = q[1]
        t = q[2]
        theta1 = q[3]
        theta2 = q[4]
        (x0, y0), (x1, y1), (x2, y2) = self._forward_kinematic_links(q)

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

        # heading arrow
        arrow_length = self.ml / 2
        dx = arrow_length * np.cos(t)
        dy = arrow_length * np.sin(t)
        arrow = mpatches.FancyArrowPatch(
            (x, y), (x + dx, y + dy), mutation_scale=20, color="blue", linewidth=2
        )
        ax.add_patch(arrow)

        # arm
        ax.plot([x0, x1, x2], [y0, y1, y2], "-o", color="red", lw=2, markersize=4)

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
        theta1 = q[3]
        theta2 = q[4]
        (x0, y0), (x1, y1), (x2, y2) = self._forward_kinematic_links(q)

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
            [x0, x1, x2], [y0, y1, y2], "-o", color="red", lw=2, markersize=4
        )

        def update(i):
            q = Q[i]
            x = q[0]
            y = q[1]
            t = q[2]
            theta1 = q[3]
            theta2 = q[4]
            (x0, y0), (x1, y1), (x2, y2) = self._forward_kinematic_links(q)

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
            linearm.set_data([x0, x1, x2], [y0, y1, y2])

        ani = FuncAnimation(fig, update, frames=len(Q), interval=100)
        plt.show()

    def _jacobian_symbolic(self):
        x, y, t, theta1, theta2 = sp.symbols("x y t theta1 theta2")
        l1, l2 = sp.symbols("l1 l2")

        xee = x + l1 * sp.cos(theta1 + t) + l2 * sp.cos(theta1 + theta2 + t)
        yee = y + l1 * sp.sin(theta1 + t) + l2 * sp.sin(theta1 + theta2 + t)

        pos = sp.Matrix([xee, yee])
        var = sp.Matrix([x, y, t, theta1, theta2])

        J = pos.jacobian(var)
        J.simplify()

        print("Jac")
        sp.pprint(J, use_unicode=True)

    def _jacobian_symbolic2(self):
        x, y, t, theta1, theta2 = sp.symbols("x y t theta1 theta2")
        l1, l2 = sp.symbols("l1 l2")

        xee = x + l1 * sp.cos(theta1 + t) + l2 * sp.cos(theta1 + theta2 + t)
        yee = y + l1 * sp.sin(theta1 + t) + l2 * sp.sin(theta1 + theta2 + t)
        tee = t + theta1 + theta2

        pos = sp.Matrix([xee, yee, tee])
        var = sp.Matrix([x, y, t, theta1, theta2])

        J = pos.jacobian(var)
        J.simplify()

        print("Jac2")
        sp.pprint(J, use_unicode=True)


if __name__ == "__main__":
    robot = MobileManipulator2d(0.5, 0.5)

    q0 = np.array([0.0, 0.0, 0.0, 1.1, -0.5])
    x = robot.forward_kinematic(q0)
    print(x)

    qdot = np.array([0.1, 0.0, 0.0, 0.0])
    J = robot.jacobian_whole_body(q0)
    print("J", J)

    xdot = J @ qdot
    print("xdot", xdot)

    robot.plot_robot(q0)

    q0 = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
    qe = np.array([1.0, 1.0, -np.pi / 2, 1.0, -1.0])
    Q = np.linspace(q0, qe, 100)
    robot.plot_robot_animation(Q)

    target = np.array([2, 2])
    q0 = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
    success, qsol = robot.inverse_kinematic(target, q0)
    print("IK success", success)
    print("qsol", qsol)

    robot.plot_robot(qsol)

    Q = np.linspace(q0, qsol, 100)
    robot.plot_robot_animation(Q)

    target2 = np.array([2, 2, np.pi / 2])
    success2, qsol2 = robot.inverse_kinematic2(target2, q0)
    print("IK2 success", success2)
    print("qsol2", qsol2)

    Q2 = np.linspace(q0, qsol2, 100)
    robot.plot_robot_animation(Q2)
