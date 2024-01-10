"""
Spline Path and Trajectory Interpolation for Navigation
Waypoint will all be pass through by interpolation unlike bspline

Type:
- "linear"    C0 (Linear spline)
- "quadratic" C0 & C1 (Quadratic spline)
- "cubic"     C0 & C1 & C2 (Cubic spline)

Reference:
- https://atsushisakai.github.io/PythonRobotics/modules/path_planning/cubic_spline/cubic_spline.html

"""
import os
import sys
wd = os.path.abspath(os.getcwd())
sys.path.append(str(wd))

import numpy as np
import bisect
from scipy import interpolate


class PolynomialSpline1D:

    def __init__(self, t, x, interpolateMode):
        self.interpolateMode = interpolateMode
        self.sx = interpolate.interp1d(t, x, kind=self.interpolateMode)

    def calculate_position(self, t):
        x = self.sx(t)
        return x


class PolynomialSpline2D:

    def __init__(self, path, interpolateMode):
        self.interpolateMode = interpolateMode
        x = path[:, 0]
        y = path[:, 1]
        self.s = self.__calc_s(x, y)
        self.sx = interpolate.interp1d(self.s, x, kind=self.interpolateMode)
        self.sy = interpolate.interp1d(self.s, y, kind=self.interpolateMode)

    def __calc_s(self, x, y):
        self.ds = np.hypot(np.diff(x), np.diff(y))
        s = [0.0]
        s.extend(np.cumsum(self.ds))
        return s

    def calculate_position(self, s):
        x = self.sx(s)
        y = self.sy(s)
        return x, y


class CubicSpline1D:

    def __init__(self, x, y):

        h = np.diff(x)
        if np.any(h < 0):
            raise ValueError("x coordinates must be sorted in ascending order")

        self.a, self.b, self.c, self.d = [], [], [], []
        self.x = x
        self.y = y
        self.nx = len(x)  # dimension of x

        # calc coefficient a
        self.a = [iy for iy in y]

        # calc coefficient c
        A = self.__calc_A(h)
        B = self.__calc_B(h, self.a)
        self.c = np.linalg.solve(A, B)

        # calc spline coefficient b and d
        for i in range(self.nx - 1):
            d = (self.c[i + 1] - self.c[i]) / (3.0 * h[i])
            b = 1.0 / h[i] * (self.a[i + 1] - self.a[i]) - h[i] / 3.0 * (2.0 * self.c[i] + self.c[i + 1])
            self.d.append(d)
            self.b.append(b)

    def calculate_position(self, x):
        if x < self.x[0]:
            return None
        elif x > self.x[-1]:
            return None

        i = self.__search_index(x)
        dx = x - self.x[i]
        position = self.a[i] + self.b[i] * dx + self.c[i] * dx**2.0 + self.d[i] * dx**3.0
        return position

    def calculate_first_derivative(self, x):
        if x < self.x[0]:
            return None
        elif x > self.x[-1]:
            return None

        i = self.__search_index(x)
        dx = x - self.x[i]
        dy = self.b[i] + 2.0 * self.c[i] * dx + 3.0 * self.d[i] * dx**2.0
        return dy

    def calculate_second_derivative(self, x):
        if x < self.x[0]:
            return None
        elif x > self.x[-1]:
            return None

        i = self.__search_index(x)
        dx = x - self.x[i]
        ddy = 2.0 * self.c[i] + 6.0 * self.d[i] * dx
        return ddy

    def __search_index(self, x):
        return bisect.bisect(self.x, x) - 1

    def __calc_A(self, h):
        A = np.zeros((self.nx, self.nx))
        A[0, 0] = 1.0
        for i in range(self.nx - 1):
            if i != (self.nx - 2):
                A[i + 1, i + 1] = 2.0 * (h[i] + h[i + 1])
            A[i + 1, i] = h[i]
            A[i, i + 1] = h[i]

        A[0, 1] = 0.0
        A[self.nx - 1, self.nx - 2] = 0.0
        A[self.nx - 1, self.nx - 1] = 1.0
        return A

    def __calc_B(self, h, a):
        B = np.zeros(self.nx)
        for i in range(self.nx - 2):
            B[i + 1] = 3.0 * (a[i + 2] - a[i + 1]) / h[i + 1] - 3.0 * (a[i + 1] - a[i]) / h[i]
        return B


class CubicSpline2D:

    def __init__(self, path):
        x = path[:, 0]
        y = path[:, 1]
        self.s = self.__calc_s(x, y)
        self.sx = CubicSpline1D(self.s, x)
        self.sy = CubicSpline1D(self.s, y)

    def __calc_s(self, x, y):
        dx = np.diff(x)
        dy = np.diff(y)
        self.ds = np.hypot(dx, dy)
        s = [0]
        s.extend(np.cumsum(self.ds))
        return s

    def calculate_position(self, s):
        x = self.sx.calculate_position(s)
        y = self.sy.calculate_position(s)
        return x, y

    def calculate_velocity(self, s):
        dx = self.sx.calculate_first_derivative(s)
        dy = self.sy.calculate_first_derivative(s)
        return dx, dy

    def calculate_acceleration(self, s):
        ddx = self.sx.calculate_second_derivative(s)
        ddy = self.sy.calculate_second_derivative(s)
        return ddx, ddy

    def calculate_curvature(self, s):
        dx, dy = self.calculate_velocity(s)
        ddx, ddy = self.calculate_acceleration(s)
        k = (ddy*dx - ddx*dy) / ((dx**2 + dy**2)**(3 / 2))
        return k

    def calculate_yaw(self, s):
        dx, dy = self.calculate_velocity(s)
        yaw = np.arctan2(dy, dx)
        return yaw

    def course(self, ds=0.1):
        s = list(np.arange(0, self.s[-1], ds))

        rx, ry, ryaw, rk = [], [], [], []
        for i_s in s:
            ix, iy = self.calculate_position(i_s)
            rx.append(ix)
            ry.append(iy)
            ryaw.append(self.calculate_yaw(i_s))
            rk.append(self.calculate_curvature(i_s))

        return rx, ry, ryaw, rk, s


if __name__ == "__main__":
    from datasave.joint_value.pre_record_value import PreRecordedPathMobileRobot
    import matplotlib.pyplot as plt

    # cubic spline 2D interplate
    path = np.array(PreRecordedPathMobileRobot.warehouse_path)
    ds = 0.01  # [m] distance of each interpolated points
    sp = CubicSpline2D(path)
    rx, ry, ryaw, rk, s = sp.course(ds)
    fig1, axis = plt.subplots(nrows=1, ncols=1)
    axis.plot(path[:, 0], path[:, 1], "xb", label="Data points")
    axis.plot(rx, ry, "-r", label="Cubic spline path")
    axis.grid(True)
    axis.axis("equal")
    axis.set_xlabel("x[m]")
    axis.set_ylabel("y[m]")
    axis.legend()

    fig2, axes2 = plt.subplots(nrows=4, ncols=1)
    axes2[0].plot(s, rx, "-r", label="Cubic spline path")
    axes2[0].set_xlabel("index")
    axes2[0].set_ylabel("x coordinate")
    axes2[1].plot(s, ry, "-r", label="Cubic spline path")
    axes2[1].set_xlabel("index")
    axes2[1].set_ylabel("y coordinate")
    axes2[2].plot(s, [np.rad2deg(iyaw) for iyaw in ryaw], "-r", label="yaw")
    axes2[2].set_xlabel("line length[m]")
    axes2[2].set_ylabel("yaw angle[deg]")
    axes2[3].plot(s, rk, "-r", label="curvature")
    axes2[3].set_xlabel("line length[m]")
    axes2[3].set_ylabel("curvature [1/m]")
    plt.show()

    # cubic spline 1D interplate
    x = np.array([0, 6, -1, 6, 0, 6, 0, 6, 0, 6, 0, 6, 0])
    time = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
    t = np.linspace(0, 11.9999, 100)
    ds = 0.01  # [m] distance of each interpolated points
    spx = CubicSpline1D(time, x)
    pos = [spx.calculate_position(ti) for ti in t]
    vel = [spx.calculate_first_derivative(ti) for ti in t]
    acc = [spx.calculate_second_derivative(ti) for ti in t]

    fig3, axis = plt.subplots(nrows=1, ncols=1)
    axis.plot(time, x, "xr", label="Discret X coordinate")
    axis.plot(t, pos, "b", label="Cubic spline path")
    axis.plot(t, vel, "-y", label="velocity")
    axis.plot(t, acc, "g", label="acceleration")
    axis.legend()
    plt.show()

    # spline using scipy
    ds = 0.1  # [m] distance of each interpolated points
    rx = []
    ry = []
    sp = PolynomialSpline2D(path, interpolateMode="quadratic")
    s = np.arange(0, sp.s[-1], ds)
    for i_s in s:
        ix, iy = sp.calculate_position(i_s)
        rx.append(ix)
        ry.append(iy)

    plt.subplots(1)
    plt.plot(path[:, 0], path[:, 1], "xr", label="Data points")
    plt.plot(rx, ry, "-")
    plt.grid(True)
    plt.axis("equal")
    plt.xlabel("x[m]")
    plt.ylabel("y[m]")
    plt.legend()
    plt.show()


    print("CubicSpline1D test")
    t = [0  ,  1, 2,   3,   4, 5, 6,  7, 8, 9, 10, 11, 12] # desired time for joint to arrive
    y = [1.7, -6, 5, 6.5, 0.0,  5,  7,  -1,  4,  5,  7, 12, 34] # desired joint position
    sp = CubicSpline1D(t, y)
    ti = np.linspace(0.0, 11.99999, 1000)
    pos = [sp.calculate_position(t) for t in ti]
    vel = [sp.calculate_first_derivative(t) for t in ti]
    acc = [sp.calculate_second_derivative(t) for t in ti]
    plt.plot(t, y, "xb", label="Data points")
    plt.plot(ti, pos, "r",label="Joint Position Cubic spline interpolation")
    plt.plot(ti, vel, "b",label="Joint Velocity")
    plt.plot(ti, acc, "g",label="Joint Accerleration")
    plt.grid(True)
    plt.legend()
    plt.show()