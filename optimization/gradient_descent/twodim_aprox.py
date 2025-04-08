import numpy as np
import matplotlib.pyplot as plt


def f(x, y):
    return x**2 + y**2


def numerical_gradient(f, x, y, h=1e-5):
    df_dx = (f(x + h, y) - f(x - h, y)) / (2 * h)
    df_dy = (f(x, y + h) - f(x, y - h)) / (2 * h)
    return np.array([df_dx, df_dy])


def analytical_gradient(f, x, y):
    return np.array([2 * x, 2 * y])


x_vals = np.linspace(-3, 3, 20)
y_vals = np.linspace(-3, 3, 20)
X, Y = np.meshgrid(x_vals, y_vals)

Uaprox = np.zeros_like(X)
Vaprox = np.zeros_like(Y)
for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        grad = numerical_gradient(f, X[i, j], Y[i, j])
        Uaprox[i, j] = grad[0]
        Vaprox[i, j] = grad[1]

# Plot the function contours and the gradient vectors
fig, ax = plt.subplots()
ax.contour(X, Y, f(X, Y), levels=20, cmap="viridis")
ax.quiver(X, Y, Uaprox, Vaprox, color="red")
ax.set_title("Numerical Gradient of $f(x, y) = x^2 + y^2$")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.grid(True)

Uanalys = np.zeros_like(X)
Vanalys = np.zeros_like(Y)
for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        grad = numerical_gradient(f, X[i, j], Y[i, j])
        Uanalys[i, j] = grad[0]
        Vanalys[i, j] = grad[1]

fig1, ax1 = plt.subplots()
ax1.contour(X, Y, f(X, Y), levels=20, cmap="viridis")
ax1.quiver(X, Y, Uanalys, Vanalys, color="red")
ax1.set_title("Analytical Gradient of $f(x, y) = x^2 + y^2$")
ax1.set_xlabel("x")
ax1.set_ylabel("y")
ax1.grid(True)

plt.show()