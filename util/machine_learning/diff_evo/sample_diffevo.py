import numpy as np
from scipy.optimize import differential_evolution
import matplotlib.pyplot as plt

# Define the objective function to minimize
def objective_function(params):
    x = np.linspace(-10, 10, 100)
    y_true = np.sin(x)
    y_pred = np.sin(params[0]*x + params[1])
    error = np.sum((y_true - y_pred)**2)
    return error

# Set the bounds for the parameters
bounds = [(0.5, 1.5), (-5, 5)]

# Run the Differential Evolution algorithm
result = differential_evolution(objective_function, bounds)

# Print the optimized parameter values and the corresponding objective function value
print('Optimized parameters:', result.x)
print('Objective function value:', result.fun)

# Define the x values
x = np.linspace(-10, 10, 100)

# Define the true sine function
y_true = np.sin(x)

# Define the predicted sine function with the optimized parameters
params = result.x
print("==>> params: \n", params)
y_pred = np.sin(params[0]*x + params[1])

# Plot the true and predicted sine functions
plt.plot(x, y_true, label='True')
plt.plot(x, y_pred, label='Predicted')
plt.legend()
plt.xlabel('x')
plt.ylabel('y')
plt.show()

# find the error between the real and the pred
error = y_true - y_pred

# plot the error
plt.plot(x, error, label="Error")
plt.legend()
plt.show()