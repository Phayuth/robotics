import numpy as np
import matplotlib.pyplot as plt

# Create a figure
fig = plt.figure()  # an empty figure with no Axes
fig, ax = plt.subplots()  # a figure with a single Axes
fig, axs = plt.subplots(2, 2)  # a figure with a 2x2 grid of Axes

# Plotting is operate on axs
t = np.linspace(0, 10, 100)
y = t**2
ax.plot(t, y, color="red", linestyle="--", linewidth=5, marker="o", label="f(t)")
ax.legend()

# plot on subplot
axs[0, 0].scatter(np.random.uniform(0, 1, 100), np.random.uniform(0, 1, 100))
axs[0, 1].contour(np.random.uniform(0, 1, (8, 8)))

# label
fig.suptitle("Figure name")
ax.set_xlabel("Time S")
ax.set_ylabel("f(t)")

plt.show()