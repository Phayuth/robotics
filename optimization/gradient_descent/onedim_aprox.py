import numpy as np
import matplotlib.pyplot as plt


def f(x):
    return x**2


def numerical_gradient(f, x, h=1e-5):
    df_dx = (f(x + h) - f(x - h)) / (2 * h)
    return df_dx


def analytical_gradient(f, x):
    return 2 * x


x = np.linspace(-3, 3, 100)
y = f(x)
grad_numerical = [numerical_gradient(f, x_val) for x_val in x]
grad_analytical = [analytical_gradient(f, x_val) for x_val in x]

fig, ax = plt.subplots()
ax.plot(x, y, label="$f(x) = x^2$")
ax.plot(x, grad_numerical, label="Numerical Gradient")
ax.plot(x, grad_analytical, label="Analytical Gradient")
ax.legend()
ax.grid(True)
plt.show()