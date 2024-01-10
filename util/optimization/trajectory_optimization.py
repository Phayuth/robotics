import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt


def cost(x):
    return x[0]


def constraints(x, N, D):
    T = x[0]
    t = np.linspace(0, T, N + 1)
    dt = t[1] - t[0]
    pos = x[1:1 + N + 1]
    vel = x[1 + (N+1):1 + (N+1) + N + 1]
    u = x[1 + 2 * (N+1):1 + 2 * (N+1) + N + 1]

    difference_pos = pos[1:] - pos[:-1] - vel[:-1] * dt  # from differential equation, we try to set it to zero = 0
    difference_vel = vel[1:] - vel[:-1] - 0.5 * (u[:-1] + u[1:]) * dt

    const = np.concatenate([difference_pos, difference_vel, [pos[0], vel[0], pos[-1] - D, vel[-1]]])
    return const


np.random.seed(1)  # random number generator seed

D = 5  # total distance
N = 10  # number of grid points

# bound
T_min, T_max = 0, 4
pos_min, pos_max = 0, D
vel_min, vel_max = -100, 100
u_min, u_max = -5, 5
bounds = [(T_min, T_max)] + [(pos_min, pos_max)] * (N+1) + [(vel_min, vel_max)] * (N+1) + [(u_min, u_max)] * (N+1)

# guess
T_opt = np.array([1])
pos_opt = np.zeros(N + 1)
vel_opt = np.zeros(N + 1)
u_opt = np.zeros(N + 1)
initial_guess = np.concatenate([T_opt, pos_opt, vel_opt, u_opt])

# test
dd = constraints(initial_guess, N, D)
print(f"==>> dd: {dd}")

result = minimize(cost, initial_guess, method='SLSQP', bounds=bounds, constraints={'type': 'eq', 'fun': lambda x: constraints(x, N, D)})
print(f"==>> result: {result}")

T = result.x[0]
pos = result.x[1:1 + N + 1]
vel = result.x[1 + N + 1:1 + (N+1) + N + 1]
u = result.x[1 + 2 * (N+1):1 + 2 * (N+1) + N + 1]

print("Optimal Time:", T)
print("Optimal Position:", pos)
print("Optimal Velocity:", vel)
print("Optimal Control:", u)


t = np.linspace(0, T, N + 1)

plt.subplot(3, 1, 1)
plt.plot(t, pos)
plt.ylabel('Position')

plt.subplot(3, 1, 2)
plt.plot(t, vel)
plt.ylabel('Velocity')

plt.subplot(3, 1, 3)
plt.plot(t, u)
plt.ylabel('Control')

plt.show()
