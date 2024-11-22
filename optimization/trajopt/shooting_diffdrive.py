import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import time


def linear_and_angular_velo_opt():
    # Constants
    dt = 0.01  # Time step
    x_initial = 1
    y_initial = 5
    theta_initial = 2

    x_final = 5
    y_final = 10

    def diffdrive_model(state, linear_velocity, angular_velocity):
        x, y, theta = state
        x_next = x + linear_velocity * np.cos(theta) * dt
        y_next = y + linear_velocity * np.sin(theta) * dt
        theta_next = theta + angular_velocity * dt
        return np.array([x_next, y_next, theta_next])

    def trajectory_positions(velocities):
        state = np.array([x_initial, y_initial, theta_initial])  # Initial state
        positions = [state[:2]]

        for i in range(velocities.shape[1]):
            state = diffdrive_model(state, velocities[0, i], velocities[1, i])
            positions.append(state[:2])

        return positions

    def objective_function(velocities):
        v = velocities.reshape(2, -1)
        positions = trajectory_positions(v)
        final_position = positions[-1]
        final_position_cost = np.linalg.norm(final_position - [x_final, y_final])
        return final_position_cost

    # Initial guess for angular velocities
    num_steps = 50
    initial_guess = np.zeros(num_steps * 2)
    times = np.linspace(0, dt * num_steps, num_steps)

    initial_time = time.time_ns()

    # Solve the optimization problem
    result = minimize(objective_function, initial_guess, method="SLSQP", options={"disp": True})

    final_time = time.time_ns()

    print("Total Time = ", (final_time - initial_time) / (10**9))

    # Extract optimal angular velocities and positions
    optimal_velocities = result.x
    optimal_velocities = optimal_velocities.reshape(2, -1)
    optimal_positions = trajectory_positions(optimal_velocities)

    # Plot the trajectory
    x_values = [pos[0] for pos in optimal_positions]
    y_values = [pos[1] for pos in optimal_positions]

    plt.figure(figsize=(8, 6))
    plt.plot(x_values, y_values, linestyle="-", color="b")
    plt.scatter([x_initial], [y_initial], color="g", label="Start")
    plt.scatter([x_final], [y_final], color="r", label="End")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.grid()
    plt.axis("equal")
    plt.show()


def linear_and_angular_velo_all_final_specified():
    # Constants
    dt = 0.01  # Time step
    x_initial = 1
    y_initial = 5
    theta_initial = 2
    x_final = 5
    y_final = 10
    theta_final = -1

    def diffdrive_model(state, linear_velocity, angular_velocity):
        x, y, theta = state
        x_next = x + linear_velocity * np.cos(theta) * dt
        y_next = y + linear_velocity * np.sin(theta) * dt
        theta_next = theta + angular_velocity * dt
        return np.array([x_next, y_next, theta_next])

    def trajectory_poses(velocities):
        state = np.array([x_initial, y_initial, theta_initial])  # Initial state
        poses = [state]

        for i in range(velocities.shape[1]):
            state = diffdrive_model(state, velocities[0, i], velocities[1, i])
            poses.append(state)

        return poses

    def objective_function(velocities):
        v = velocities.reshape(2, -1)
        poses = trajectory_poses(v)
        final_pose = poses[-1]
        terminal_position_cost = np.linalg.norm(final_pose[:2] - [x_final, y_final])
        terminal_orientation_cost = np.linalg.norm(final_pose[-1] - theta_final)
        final_cost = terminal_position_cost + terminal_orientation_cost
        return final_cost

    # Initial guess for angular velocities
    num_steps = 50
    initial_guess = np.zeros(num_steps * 2)
    times = np.linspace(0, dt * num_steps, num_steps)

    initial_time = time.time_ns()

    # Solve the optimization problem
    result = minimize(objective_function, initial_guess, method="SLSQP", options={"disp": True})

    final_time = time.time_ns()

    print("Total Time = ", (final_time - initial_time) / (10**9))

    # Extract optimal angular velocities and positions
    optimal_velocities = result.x
    optimal_velocities = optimal_velocities.reshape(2, -1)
    optimal_poses = trajectory_poses(optimal_velocities)

    # Plot the trajectory
    x_values = [pos[0] for pos in optimal_poses]
    y_values = [pos[1] for pos in optimal_poses]

    plt.figure(figsize=(8, 6))
    plt.plot(x_values, y_values, marker="*", linestyle="-", color="b")
    plt.scatter([x_initial], [y_initial], color="g", label="Start")
    plt.scatter([x_final], [y_final], color="r", label="End")
    plt.quiver(x_initial, y_initial, 1*np.cos(theta_initial), 1*np.sin(theta_initial), color="g")
    plt.quiver(x_final, y_final, 1*np.cos(theta_final), 1*np.sin(theta_final), color="r")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.grid()
    plt.axis("equal")
    plt.show()



if __name__ == "__main__":
    # linear_and_angular_velo_opt()
    linear_and_angular_velo_all_final_specified()
