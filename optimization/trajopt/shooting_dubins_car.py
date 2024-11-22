import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import time


def angular_velo_opt():
    # Constants
    linear_velocity = 1.0
    turn_radius = 1

    dt = 0.01  # Time step
    x_initial = 1
    y_initial = 5
    theta_initial = 0

    x_final = 5
    y_final = 10

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
    result = minimize(objective_function, initial_guess, method="SLSQP", options={"disp": True})

    final_time = time.time_ns()

    print("Total Time = ", (final_time - initial_time) / (10**9))

    # Extract optimal angular velocities and positions
    optimal_angular_velocities = result.x
    print(f"> type(optimal_angular_velocities): {type(optimal_angular_velocities)}")
    print(f"> optimal_angular_velocities_len): {len(optimal_angular_velocities)}")
    print(f"> optimal_angular_velocities: {optimal_angular_velocities}")
    optimal_positions = trajectory_positions(optimal_angular_velocities)
    print(f"> optimal_positions: {optimal_positions}")

    # Plot the trajectory
    x_values = [pos[0] for pos in optimal_positions]
    y_values = [pos[1] for pos in optimal_positions]

    plt.figure(figsize=(8, 6))
    plt.plot(x_values, y_values, linestyle="-", color="b")
    plt.scatter([x_initial], [y_initial], color="g", label="Start")
    plt.scatter([x_final], [y_final], color="r", label="End")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Dubin's Car Path")
    plt.legend()
    plt.grid()
    plt.axis("equal")
    plt.show()


def linear_and_angular_velo_opt():
    # Constants
    dt = 0.01  # Time step
    x_initial = 1
    y_initial = 5
    theta_initial = -np.pi

    x_final = 5
    y_final = 10

    def dubins_car_model(state, linear_velocity, angular_velocity):
        turn_radius = 1
        x, y, theta = state
        x_next = x + linear_velocity * np.cos(theta) * dt
        y_next = y + linear_velocity * np.sin(theta) * dt
        theta_next = theta + (linear_velocity * np.tan(angular_velocity) / turn_radius) * dt
        return np.array([x_next, y_next, theta_next])

    def trajectory_positions(velocities):
        state = np.array([x_initial, y_initial, theta_initial])  # Initial state
        positions = [state[:2]]

        for i in range(velocities.shape[1]):
            state = dubins_car_model(state, velocities[0, i], velocities[1, i])
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

    state = np.array([x_initial, y_initial, theta_initial])
    positions = [state[:2]]
    thetas = [state[2]]
    for i in range(optimal_velocities.shape[1]):
        state = dubins_car_model(state, optimal_velocities[0, i], optimal_velocities[1, i])
        positions.append(state[:2])
        thetas.append(state[2])

    # Plot the trajectory
    x_values = [pos[0] for pos in positions]
    y_values = [pos[1] for pos in positions]

    print(len(x_values), len(y_values), len(thetas), len(times))

    plt.figure(figsize=(8, 6))
    plt.plot(x_values, y_values, linestyle="-", color="b")
    plt.scatter([x_initial], [y_initial], color="g", label="Start")
    plt.scatter([x_final], [y_final], color="r", label="End")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Dubin's Car Path")
    plt.legend()
    plt.grid()
    plt.axis("equal")

    timessssssss = np.linspace(0, dt * num_steps, num_steps + 1)
    print(len(timessssssss))

    fig, axs = plt.subplots(3, 1, figsize=(10, 15), sharex=True)

    axs[0].plot(timessssssss, thetas, label=f"Theta")
    axs[0].grid(True)
    axs[0].legend(loc="upper right")

    axs[1].plot(times, optimal_velocities[0], label=f"Linear Velo")
    axs[1].grid(True)
    axs[1].legend(loc="upper right")

    axs[2].plot(times, optimal_velocities[1], label=f"Angular Velo")
    axs[2].grid(True)
    axs[2].legend(loc="upper right")

    axs[-1].set_xlabel("Time")
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    plt.show()


if __name__=="__main__":
    # angular_velo_opt()
    linear_and_angular_velo_opt()
