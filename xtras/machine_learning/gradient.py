import numpy as np
import matplotlib.pyplot as plt


def numerical_gradient(f, x, h=1e-5):
    grad = np.zeros_like(x)  # Initialize the gradient vector
    for i in range(len(x)):
        x_plus = np.copy(x)
        x_minus = np.copy(x)
        x_plus[i] += h
        x_minus[i] -= h
        grad[i] = (f(x_plus) - f(x_minus)) / (2 * h)
    return grad


# Example function: f(x, y) = x^2 + y^2
def f(x):
    return x[0] ** 2 + x[1] ** 2


def f(x, y):
    return x**2 + y**2


# Compute gradient at (1, 2)
point = np.array([1.0, 2.0])
grad = numerical_gradient(f, point)
print("Numerical Gradient:", grad)


# Define the numerical gradient function
def numerical_gradient(f, x, y, h=1e-5):
    df_dx = (f(x + h, y) - f(x - h, y)) / (2 * h)
    df_dy = (f(x, y + h) - f(x, y - h)) / (2 * h)
    return np.array([df_dx, df_dy])


# Create a grid of points for plotting
x_vals = np.linspace(-3, 3, 20)
y_vals = np.linspace(-3, 3, 20)
X, Y = np.meshgrid(x_vals, y_vals)

# Compute the gradient at each point on the grid
U = np.zeros_like(X)
V = np.zeros_like(Y)
for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        grad = numerical_gradient(f, X[i, j], Y[i, j])
        U[i, j] = grad[0]
        V[i, j] = grad[1]

# Plot the function contours and the gradient vectors
plt.figure(figsize=(8, 8))
plt.contour(X, Y, f(X, Y), levels=20, cmap="viridis")
plt.quiver(X, Y, U, V, color="red")
plt.title("Numerical Gradient of $f(x, y) = x^2 + y^2$")
plt.xlabel("x")
plt.ylabel("y")
plt.grid(True)
plt.show()
