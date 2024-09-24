import numpy as np
from scipy.optimize import minimize

g = 9.81  # Gravity (m/s^2)
L = 1.0   # Length of the pendulum (m)
N = 50
T = 2.0  # Total time (s)
dt = T / N

theta0 = 0      # Starting upside down
thetaf = np.pi  # Desired final position
omega0 = 0      # Initial angular velocity
omegaf = 0      # Final angular velocity

def cost(u):
    return np.sum(u**2)

def dynamics(u, dt, N):
    theta = np.zeros(N+1)
    omega = np.zeros(N+1)
    theta[0] = theta0
    omega[0] = omega0

    for i in range(N):
        omega[i+1] = omega[i] + (-g/L * np.sin(theta[i]) + u[i]) * dt
        theta[i+1] = theta[i] + omega[i+1] * dt

    return theta, omega

def constraints(u):
    theta, omega = dynamics(u, dt, N)
    return [theta[-1] - thetaf, omega[-1] - omegaf]

u0 = np.zeros(N)
cons = {'type': 'eq', 'fun': constraints}
result = minimize(cost, u0, constraints=cons, method='SLSQP')

u_opt = result.x
theta_opt, omega_opt = dynamics(u_opt, dt, N)

import matplotlib.pyplot as plt

time = np.linspace(0, T, N+1)
plt.figure(figsize=(10, 5))

plt.subplot(3, 1, 1)
plt.plot(time, theta_opt)
plt.xlabel('Time (s)')
plt.ylabel('Theta (rad)')
plt.title('Theta vs Time')

plt.subplot(3, 1, 2)
plt.plot(time, omega_opt)
plt.xlabel('Time (s)')
plt.ylabel('Omega (rad/s)')
plt.title('Omega vs Time')

plt.subplot(3, 1, 3)
plt.plot(time[:-1], u_opt)
plt.xlabel('Time (s)')
plt.ylabel('Control Input (u)')
plt.title('Control Input vs Time')

plt.tight_layout()
plt.show()
