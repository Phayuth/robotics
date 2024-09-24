import casadi as ca

# Rosenbrock function
x = ca.SX.sym("x")
y = ca.SX.sym("y")

a = 1
b = 100
rosenbrock = (a - x) ** 2 + b * (y - x**2) ** 2

# Create an optimization problem
nlp = {
    "x": ca.vertcat(x, y),
    "f": rosenbrock,
}

# Create an NLP solver
opts = {
    "ipopt.print_level": 0,
    "print_time": 0,
}
solver = ca.nlpsol("solver", "ipopt", nlp, opts)

sol = solver(x0=[0, 0], lbg=-ca.inf, ubg=ca.inf)
x_opt = sol["x"].full().flatten()

print(f"Optimal solution: x = {x_opt[0]}, y = {x_opt[1]}")


import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(-10, 10, 500)
y = np.linspace(-10, 10, 500)
X, Y = np.meshgrid(x, y)
Z = (a - X) ** 2 + b * (Y - X**2) ** 2

fig, ax = plt.subplots()
CS = ax.contour(X, Y, Z)
ax.clabel(CS, inline=True, fontsize=10)
ax.plot([0], [1], "r*")
ax.plot(x_opt[0], x_opt[1], "g*")
plt.show()
