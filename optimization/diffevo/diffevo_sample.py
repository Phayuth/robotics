import numpy as np
from scipy.optimize import differential_evolution
import matplotlib.pyplot as plt


def objective_function(params):
    x = np.linspace(-10, 10, 100)
    y_true = np.sin(x)
    y_pred = np.sin(params[0] * x + params[1])
    error = np.sum((y_true - y_pred) ** 2)
    return error


bounds = [(0.5, 1.5), (-5, 5)]
result = differential_evolution(objective_function, bounds)
print("Optimized parameters:", result.x)
print("Objective function value:", result.fun)


x = np.linspace(-10, 10, 100)
y_true = np.sin(x)
params = result.x
print("==>> params: \n", params)
y_pred = np.sin(params[0] * x + params[1])
error = y_true - y_pred


plt.plot(x, y_true, label="True")
plt.plot(x, y_pred, label="Predicted")
plt.legend()
plt.xlabel("x")
plt.ylabel("y")
plt.show()
plt.plot(x, error, label="Error")
plt.legend()
plt.show()
