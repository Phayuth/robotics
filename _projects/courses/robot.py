import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RadioButtons
from matplotlib.animation import FuncAnimation


class Robot2Joints:

    def __init__(self, l1, l2):
        self.l1 = l1
        self.l2 = l2

    def forward_kinematics(self, theta1, theta2):
        x = self.l1 * np.cos(theta1) + self.l2 * np.cos(theta1 + theta2)
        y = self.l1 * np.sin(theta1) + self.l2 * np.sin(theta1 + theta2)
        return np.array([x, y]).reshape(2, 1)

    def inverse_kinematics(self, x, y, elbow_option):
        rd = np.sqrt(x**2 + y**2)
        link_length = self.l1 + self.l2
        if rd > link_length:
            print("The desired pose is outside of taskspace")
            return [0, 0]

        if elbow_option == 0:
            sign = -1
        elif elbow_option == 1:
            sign = 1

        D = (x**2 + y**2 - self.l1**2 - self.l2**2) / (2 * self.l1 * self.l2)
        theta2 = np.arctan2(sign * np.sqrt(1 - D**2), D)
        theta1 = np.arctan2(y, x) - np.arctan2(
            (self.l2 * np.sin(theta2)),
            (self.l1 + self.l2 * np.cos(theta2)),
        )
        return [theta1, theta2]

    def jacobian(self, theta1, theta2):
        J = np.array(
            [
                [
                    -self.l1 * np.sin(theta1) - self.l2 * np.sin(theta1 + theta2),
                    -self.l2 * np.sin(theta1 + theta2),
                ],
                [
                    self.l1 * np.cos(theta1) + self.l2 * np.cos(theta1 + theta2),
                    self.l2 * np.cos(theta1 + theta2),
                ],
            ]
        )
        return J

    def velocity_kinematics(self, theta1, theta2, dtheta1, dtheta2):
        J = self.jacobian(theta1, theta2)
        Xdot = J @ np.array([dtheta1, dtheta2]).reshape(2, 1)
        return Xdot

    def _fk_links(self, theta1, theta2):
        """
        !DO NOT MODIFY THIS METHOD
        """
        x0, y0 = 0, 0
        x1 = self.l1 * np.cos(theta1)
        y1 = self.l1 * np.sin(theta1)
        x2 = x1 + self.l2 * np.cos(theta1 + theta2)
        y2 = y1 + self.l2 * np.sin(theta1 + theta2)
        return (x0, x1, x2), (y0, y1, y2)

    def view_forward_kinematics(self):
        """
        !DO NOT MODIFY THIS METHOD
        """
        fig, ax = plt.subplots(figsize=(8, 6))
        plt.subplots_adjust(bottom=0.25)

        # --- Initial Plot Data ---
        theta1 = 0.0
        theta2 = 0.0
        (x, y) = self._fk_links(theta1, theta2)
        xe, ye = x[-1], y[-1]

        (line1,) = ax.plot(x, y, lw=2, color="#1f77b4")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_aspect("equal", "box")
        ax.set_title("Forward Kinematic")
        ax.grid(True)
        ax.set_ylim(-2, 2)
        ax.set_xlim(-2, 2)

        slider1_ax = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor="lightgray")
        slider2_ax = plt.axes([0.25, 0.15, 0.65, 0.03], facecolor="lightgray")

        joint1slider = Slider(
            ax=slider1_ax,
            label="Theta1 (rad)",
            valmin=-np.pi,
            valmax=np.pi,
            valinit=theta1,
            valstep=0.01,
            color="#1f77b4",
        )

        joint2slider = Slider(
            ax=slider2_ax,
            label="Theta2 (rad)",
            valmin=-np.pi,
            valmax=np.pi,
            valinit=theta2,
            valstep=0.01,
            color="#1f77b4",
        )

        def update(val):
            theta1 = joint1slider.val
            theta2 = joint2slider.val
            (x, y) = self._fk_links(theta1, theta2)
            line1.set_xdata(x)
            line1.set_ydata(y)
            fig.canvas.draw_idle()

        joint1slider.on_changed(update)
        joint2slider.on_changed(update)

        plt.show()

    def view_inverse_kinematics(self):
        """
        !DO NOT MODIFY THIS METHOD
        """
        fig, ax = plt.subplots(figsize=(8, 6))
        plt.subplots_adjust(bottom=0.25)

        # --- Initial Plot Data ---
        theta1 = 0.0
        theta2 = 0.0
        (x, y) = self._fk_links(theta1, theta2)
        xe, ye = x[-1], y[-1]

        (line1,) = ax.plot(x, y, lw=2, color="#1f77b4")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_aspect("equal", "box")
        ax.set_title("Inverse Kinematic")
        ax.grid(True)
        ax.set_ylim(-2, 2)
        ax.set_xlim(-2, 2)

        checker_ax = plt.axes([0.25, 0.1, 0.5, 0.1], facecolor="lightgray")
        checker = RadioButtons(
            ax=checker_ax,
            labels=["Elbow up", "Elbow down"],
        )

        def update(val):
            if checker.value_selected == "Elbow up":
                option = 0
            elif checker.value_selected == "Elbow down":
                option = 1

            if isinstance(val, str):
                pass
            else:
                theta1, theta2 = self.inverse_kinematics(
                    val.xdata,
                    val.ydata,
                    option,
                )
                (x, y) = self._fk_links(theta1, theta2)
                line1.set_xdata(x)
                line1.set_ydata(y)
            fig.canvas.draw_idle()

        checker.on_clicked(update)
        fig.canvas.mpl_connect("button_press_event", update)
        plt.show()

    def view_trajectory_animation(self, Q):
        """
        !DO NOT MODIFY THIS METHOD
        """
        fig, ax = plt.subplots(figsize=(8, 6))

        x, y = self._fk_links(Q[0, 0], Q[0, 1])
        (line,) = ax.plot(x, y, lw=2, color="#1f77b4")
        (trail_line,) = ax.plot(
            [],
            [],
            lw=1.5,
            color="#ff7f0e",
            alpha=0.7,
            label="End-Effector Trail",
        )

        ax.set_xlim(-2, 2)
        ax.set_ylim(-2, 2)
        ax.set_aspect("equal", "box")
        ax.grid(True)
        ax.set_title("2-Joint Robot Trajectory Animation")
        ax.legend()

        # Precompute end-effector positions for the trail
        ee_x = []
        ee_y = []
        for i in range(Q.shape[0]):
            x_, y_ = self._fk_links(Q[i, 0], Q[i, 1])
            ee_x.append(x_[-1])
            ee_y.append(y_[-1])
        ee_x = np.array(ee_x)
        ee_y = np.array(ee_y)

        def update(i):
            x, y = self._fk_links(Q[i, 0], Q[i, 1])
            line.set_xdata(x)
            line.set_ydata(y)
            # Update trail up to current frame
            trail_line.set_xdata(ee_x[: i + 1])
            trail_line.set_ydata(ee_y[: i + 1])
            return line, trail_line

        ani = FuncAnimation(fig, update, frames=Q.shape[0], interval=50, blit=True)
        plt.show()


def run_fk():
    robot = Robot2Joints(1, 1)
    theta1 = 0.0
    theta2 = 0.0
    X = robot.forward_kinematics(theta1, theta2)
    print(f"X: {X}")


def run_ik():
    robot = Robot2Joints(1, 1)
    x = 1.0
    y = 1.0
    theta1, theta2 = robot.inverse_kinematics(x, y, 0)
    print(f"Theta1: {theta1}, Theta2: {theta2}")


def run_jac():
    robot = Robot2Joints(1, 1)
    theta1 = 0.0
    theta2 = 0.0
    J = robot.jacobian(theta1, theta2)
    print(f"Jacobian: \n{J}")


def run_fv():
    robot = Robot2Joints(1, 1)
    theta1 = 0.0
    theta2 = 0.0
    dtheta1 = 0.1
    dtheta2 = 0.1
    Xdot = robot.velocity_kinematics(theta1, theta2, dtheta1, dtheta2)
    print(f"Xdot: \n{Xdot}")


def run_viewfk():
    robot = Robot2Joints(1, 1)
    robot.view_forward_kinematics()


def run_viewik():
    robot = Robot2Joints(1, 1)
    robot.view_inverse_kinematics()


def circle_trajectory(r, t):
    x = r * np.cos(t)
    y = r * np.sin(t)
    return x, y


def run_taskspace_trajectory():
    t = np.linspace(0, 2 * np.pi, 100)
    x, y = circle_trajectory(1, t)

    robot = Robot2Joints(1, 1)
    eo = 0
    Q = []
    for x, y in zip(x, y):
        theta1, theta2 = robot.inverse_kinematics(x, y, eo)
        Q.append([theta1, theta2])
    Q = np.vstack(Q)
    robot.view_trajectory_animation(Q)


if __name__ == "__main__":
    func = [
        run_fk,
        run_ik,
        run_jac,
        run_fv,
        run_viewfk,
        run_viewik,
        run_taskspace_trajectory,
    ]
    for f in func:
        f()
