import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

# x = np.linspace(0, 10, 100)
# y1 = np.sin(x)
# y2 = np.cos(x)
# fig, ax1 = plt.subplots()
# ax1.plot(x, y1, 'b-', label='y1')
# ax1.set_xlabel('X')
# ax1.set_ylabel('Y1', color='b')
# ax1.tick_params('y', colors='b')
# ax2 = ax1.twinx()
# ax2.plot(x, y2, 'r-', label='y2')
# ax2.set_ylabel('Y2', color='r')
# ax2.tick_params('y', colors='r')
# lines = ax1.get_lines() + ax2.get_lines()
# ax1.legend(lines, [line.get_label() for line in lines], loc='upper right')
# plt.show()


# fig = plt.figure()
# ax1 = fig.add_subplot(2, 1, 1)  # First subplot for the first graph
# ax2 = fig.add_subplot(2, 1, 2)  # Second subplot for the second graph
# x1 = np.linspace(0, 2 * np.pi, 100)
# y1 = np.sin(x1)
# x2 = np.linspace(0, 2 * np.pi, 100)
# y2 = np.cos(x2)
# line1, = ax1.plot([], [], 'r-', label='sin(x)')
# line2, = ax2.plot([], [], 'b-', label='cos(x)')
# ax1.set_xlim(0, 2 * np.pi)
# ax1.set_ylim(-1, 1)
# ax1.set_xlabel('x')
# ax1.set_ylabel('y')
# ax2.set_xlim(0, 2 * np.pi)
# ax2.set_ylim(-1, 1)
# ax2.set_xlabel('x')
# ax2.set_ylabel('y')
# def update(frame):
#     line1.set_data(x1[:frame], y1[:frame])  # Update data for the first line
#     line2.set_data(x2[:frame], y2[:frame])  # Update data for the second line
# animation = FuncAnimation(fig, update, frames=len(x1), interval=50)
# ax1.legend()
# ax2.legend()
# plt.show()


fig, ax = plt.subplots()
x = np.linspace(0, 2 * np.pi, 100)
y1 = np.sin(x)
line1, = ax.plot(x, y1, 'r-', label='sin(x)')
line2, = ax.plot([], [], 'b-', label='animated plot')
ax.set_xlim(0, 2 * np.pi)
ax.set_ylim(-1, 1)
ax.set_xlabel('x')
ax.set_ylabel('y')
def update(frame):
    y2 = np.cos(x + frame * 0.1)  # Update the y-values for the animated plot
    line2.set_data(x, y2)
animation = FuncAnimation(fig, update, frames=100, interval=50)
ax.legend()
plt.show()

