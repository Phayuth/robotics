import casadi as ca

x = ca.MX.sym("x")
y = ca.MX.sym("y")
objective = x**2 + y**2

optProblem = {
    "x": ca.vertcat(x, y),
    "f": objective,
}
opts = {"ipopt.print_level": 0, "print_time": 0, "ipopt.tol": 1e-6}
solver = ca.nlpsol("solver", "ipopt", optProblem, opts)
InitialGuess = [1.0, 2.0]
solution = solver(x0=InitialGuess)

print(solution["x"])
resnp = solution["x"].__array__().flatten()

import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(-10, 10, 50)
y = np.linspace(-10, 10, 50)
X, Y = np.meshgrid(x, y)
Z = X**2 + Y**2


fig, ax = plt.subplots()
CS = ax.contour(X, Y, Z)
ax.clabel(CS, inline=True, fontsize=10)
ax.plot(InitialGuess[0], InitialGuess[1], "r*")
ax.plot(resnp[0], resnp[1], "g*")
plt.show()


fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
surf = ax.plot_surface(X, Y, Z, linewidth=0, antialiased=False)
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.show()
