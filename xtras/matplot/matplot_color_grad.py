import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np

# Define a list of colors to use for the gradient
colors_list = [(0, "#FFFFFF"), (0.5, "#7F7F7F"), (1, "#000000")] # white -> gray -> black

# Create the colormap object with the colors list
cmap = colors.LinearSegmentedColormap.from_list("my_cmap", colors_list)

# Generate some sample data to plot
x = np.linspace(0, 1, 100)
y = np.sin(x)

# Create a scatter plot with the color map
plt.scatter(x, y, c=x, cmap=cmap)

# Add a color bar
plt.colorbar()

# Show the plot
plt.show()