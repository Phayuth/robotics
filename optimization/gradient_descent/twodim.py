import numpy as np
import matplotlib.pyplot as plt


class Parabola3D:

    def __init__(self, a, b, c):
        self.a = a
        self.b = b
        self.c = c

    def f(self, x, y):
        return x**2 + y**2

    def f_grad_analytical(self, x, y):
        return [2 * x, 2 * y]


# gradient descent
xinit = 4
yinit = 4
xopt = [xinit]
yopt = [yinit]
p = Parabola3D(1, -4, 5)
eta = 0.1
for i in range(100):
    xinit -= eta * p.f_grad_analytical(xinit, yinit)[0]
    yinit -= eta * p.f_grad_analytical(xinit, yinit)[1]
    xopt.append(xinit)
    yopt.append(yinit)
    print(f"iter {i}, x = {xinit}, y = {yinit}, z = {p.f(xinit, yinit)}")

# plot
x = np.arange(-5, 5, 0.1)
y = np.arange(-5, 5, 0.1)
X, Y = np.meshgrid(x, y)
Z = p.f(X, Y)
fig = plt.figure()
ax = fig.add_subplot(projection="3d")
ax.plot_surface(X, Y, Z, cmap="viridis")
ax.plot(xopt, yopt, [p.f(x, y) for x, y in zip(xopt, yopt)], "-o", markersize=10, color="r")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("f(x, y)")
plt.show()
