import numpy as np
import bisect
import matplotlib.pyplot as plt

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

    def calc_position(self, x):

        # check if the time given is inside the segment area
        if x < self.x[0]:
            return None
        elif x > self.x[-1]:
            return None

        # check if the time given belong to what segment area
        i = self.__search_index(x)
        dx = x - self.x[i]

        # calculate the equation q(t) = a + bt + ct^2 + dt^3
        position = self.a[i] + self.b[i] * dx + self.c[i] * dx ** 2.0 + self.d[i] * dx ** 3.0

        return position

    def calc_first_derivative(self, x):
     
        if x < self.x[0]:
            return None
        elif x > self.x[-1]:
            return None

        i = self.__search_index(x)
        dx = x - self.x[i]
        dy = self.b[i] + 2.0 * self.c[i] * dx + 3.0 * self.d[i] * dx ** 2.0
        return dy

    def calc_second_derivative(self, x):

        if x < self.x[0]:
            return None
        elif x > self.x[-1]:
            return None

        i = self.__search_index(x)
        dx = x - self.x[i]
        ddy = 2.0 * self.c[i] + 6.0 * self.d[i] * dx
        return ddy

    def __search_index(self, x):
        """
        search data segment index
        """
        return bisect.bisect(self.x, x) - 1

    def __calc_A(self, h):
        """
        calc matrix A for spline coefficient c
        """
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
        """
        calc matrix B for spline coefficient c
        """
        B = np.zeros(self.nx)
        for i in range(self.nx - 2):
            B[i + 1] = 3.0 * (a[i + 2] - a[i + 1]) / h[i + 1] - 3.0 * (a[i + 1] - a[i]) / h[i]
        return B

print("CubicSpline1D test") 
x = [0, 6, 0, 6, 0,  6,  0,  6,  0,  6,  0, 6, 0]
y = [6, 0, 6, 0,  6,  0,  6,  0,  6,  0, 6, 0, 0]

t = [0  ,  1, 2,   3,   4, 5, 6,  7, 8, 9, 10, 11, 12]

ds = 0.01  # [m] distance of each interpolated points

spx = CubicSpline1D(t, x)
spy = CubicSpline1D(t, y)

ti = np.linspace(0.0, 14, 1000)

rx = [spx.calc_position(t) for t in ti]
ry = [spy.calc_position(t) for t in ti]

# vel = [sp.calc_first_derivative(t) for t in ti]
# acc = [sp.calc_second_derivative(t) for t in ti]

# print(ti)
# eq = sp.calc_position(0.5)

plt.subplots(1)
plt.plot(x, y, "xb", label="Data points")
plt.plot(rx, ry, "-r", label="Cubic spline path")
plt.grid(True)
plt.axis("equal")
plt.xlabel("x[m]")
plt.ylabel("y[m]")
plt.legend()

plt.subplots(1)
plt.plot(t, y, "xb", label="Data points")
plt.plot(ti, ry, "-r", label="Cubic spline path")
plt.grid(True)
plt.axis("equal")
plt.xlabel("t[s]")
plt.ylabel("y[m]")
plt.legend()

plt.subplots(1)
plt.plot(t, x, "xb", label="Data points")
plt.plot(ti, rx, "-r", label="Cubic spline path")
plt.grid(True)
plt.axis("equal")
plt.xlabel("t[s]")
plt.ylabel("x[m]")
plt.legend()

plt.show()