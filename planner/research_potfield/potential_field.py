import matplotlib.pyplot as plt
import numpy as np

# define constants
k_att = 1.0
k_rep = 1.0
d_max = 5.0

# define goal and obstacle positions
q_goal = np.array([5, 5])  # attractive point
q_obs = np.array([3, 3])   # repulsive point (obstacle)

# create a grid for visualization
x_min, x_max = -10, 10
y_min, y_max = -10, 10
x_range = np.arange(x_min, x_max, 0.1)
y_range = np.arange(y_min, y_max, 0.1)
xx, yy = np.meshgrid(x_range, y_range)

# calculate the potential field over the grid
U_total = np.zeros_like(xx)
for i in range(len(x_range)):
    for j in range(len(y_range)):
        q = np.array([xx[i, j], yy[i, j]])
        U_total[i, j] = k_att * np.linalg.norm(q - q_goal)**2 + k_rep * (1/np.linalg.norm(q - q_obs) - 1/d_max)**2

# plot the potential field and the goal and obstacle positions
fig, ax = plt.subplots()
ax.set_xlim([x_min, x_max])
ax.set_ylim([y_min, y_max])
ax.set_aspect('equal')
ax.contourf(xx, yy, U_total, levels=np.linspace(np.min(U_total), np.max(U_total), 100))
ax.plot(q_goal[0], q_goal[1], 'go', markersize=10)
ax.plot(q_obs[0], q_obs[1], 'ro', markersize=10)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('Artificial Potential Field')
plt.show()