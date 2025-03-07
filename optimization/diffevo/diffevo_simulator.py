import numpy as np
from scipy.optimize import differential_evolution
import matplotlib.pyplot as plt


def simulator_function(params):
    x = np.linspace(-10, 10, 100)
    y_pred = params[0] * x**2 + params[1] * x + params[2]
    return y_pred


x = np.linspace(-10, 10, 100)
real_data = x**2 + x + 8


def objective_function(params):
    simulator_output = simulator_function(params)
    distance = np.sum((simulator_output - real_data) ** 2)
    return distance


bounds = [(0, 10), (0, 10), (0, 10)]
strategy = "best1bin"
popsize = 10
mutation = (0.5, 1)
recombination = 0.7


init_population = [np.random.uniform(low=b[0], high=b[1]) for b in bounds]
result = differential_evolution(objective_function, bounds)
estimated_params = result.x
print(f"==>> estimated_params: \n{estimated_params}")
simulator_output = simulator_function(estimated_params)

plt.plot(x, simulator_output)
plt.plot(x, real_data)
plt.show()
