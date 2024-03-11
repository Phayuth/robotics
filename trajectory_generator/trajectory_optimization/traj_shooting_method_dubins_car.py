import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import time

# Constants
linear_velocity = 1.0
dt = 0.01  # Time step
x_initial = 1
y_initial = 5
x_final = 5
y_final = 10
turn_radius = 1
theta_initial = 0

def dubins_car_model(state, angular_velocity):
    x, y, theta = state
    x_next = x + linear_velocity * np.cos(theta) * dt
    y_next = y + linear_velocity * np.sin(theta) * dt
    theta_next = theta + (linear_velocity * np.tan(angular_velocity) / turn_radius) * dt
    return np.array([x_next, y_next, theta_next])

def trajectory_positions(angular_velocities):
    state = np.array([x_initial, y_initial, theta_initial])  # Initial state
    positions = [state[:2]]

    for angular_velocity in angular_velocities:
        state = dubins_car_model(state, angular_velocity)
        positions.append(state[:2])

    return positions

def objective_function(angular_velocities):
    positions = trajectory_positions(angular_velocities)
    final_position = positions[-1]
    final_position_cost = np.linalg.norm(final_position - [x_final, y_final])
    return final_position_cost

# Initial guess for angular velocities
num_steps = 1000
initial_guess = np.zeros(num_steps)

initial_time = time.time_ns()

# Solve the optimization problem
result = minimize(objective_function, initial_guess, method='SLSQP', options={'disp': True})

final_time = time.time_ns()

print("Total Time = ", (final_time-initial_time)/ (10 ** 9))

# Extract optimal angular velocities and positions
optimal_angular_velocities = result.x
optimal_positions = trajectory_positions(optimal_angular_velocities)

# Plot the trajectory
x_values = [pos[0] for pos in optimal_positions]
y_values = [pos[1] for pos in optimal_positions]

plt.figure(figsize=(8, 6))
plt.plot(x_values, y_values, linestyle='-', color='b')
plt.scatter([x_initial], [y_initial], color='g', label='Start')
plt.scatter([x_final], [y_final], color='r', label='End')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Dubin\'s Car Path')
plt.legend()
plt.grid()
plt.axis('equal')
plt.show()
