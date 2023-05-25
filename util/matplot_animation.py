import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

fig, ax = plt.subplots()

# Create a line plot with initial data
x_data = np.arange(0, 10, 0.1)
y_data = np.sin(x_data)
line, = ax.plot(x_data, y_data)

# Define the update function for the animation
def update(frame):
    # Update the y data
    line.set_ydata(np.sin(x_data + frame/10))

    # Return the line object to be updated
    return line,

# Create the animation object
ani = FuncAnimation(fig, update, frames=100, interval=50, blit=True)

# Show the plot
plt.show()