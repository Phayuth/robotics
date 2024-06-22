import numpy as np
from scipy.optimize import differential_evolution
import matplotlib.pyplot as plt

def simulator_function(params):
    x = np.linspace(-10, 10, 100)
    y_pred = params[0]*x**2 + params[1]*x+ params[2]
    return y_pred

x = np.linspace(-10, 10, 100)
real_data  = x**2 + x + 8


# Define the objective function
def objective_function(params):
    # Run the simulator with the current parameter values
    simulator_output = simulator_function(params)
    
    # Calculate the distance between the simulated output and the real-world data
    distance = np.sum((simulator_output - real_data)**2)
    
    return distance

# Set up the differential evolution algorithm
bounds = [(0, 10), (0, 10), (0, 10)] # parameter value ranges
strategy = 'best1bin' # mutation strategy
popsize = 10 # population size
mutation = (0.5, 1) # mutation factor
recombination = 0.7 # crossover rate

# Initialize the population
init_population = [np.random.uniform(low=b[0], high=b[1]) for b in bounds]

# Run the differential evolution algorithm
result = differential_evolution(objective_function, bounds)

# Extract the estimated parameter values
estimated_params = result.x
print(f"==>> estimated_params: \n{estimated_params}")

# Validate the estimated parameter values
simulator_output = simulator_function(estimated_params)

plt.plot(x, simulator_output)
plt.plot(x, real_data)
plt.show()