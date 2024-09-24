import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np

# Define a list of colors to use for the gradient
colors_list = [(0, "#FFFFFF"), (0.5, "#7F7F7F"), (1, "#000000")] # white -> gray -> black
cmap = colors.LinearSegmentedColormap.from_list("my_cmap", colors_list)

x = np.linspace(0, 1, 100)
y = np.sin(x)

plt.scatter(x, y, c=x, cmap=cmap)
plt.colorbar()
plt.show()