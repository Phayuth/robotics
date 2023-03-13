import numpy as np
import glob
import matplotlib.pyplot as plt
from load_map import grid_map_binary, grid_map_probability
from dummy_map import pmap

# View map -------------------------------------------------------------------------------------
index = 2
map = grid_map_binary(index)
plt.imshow(map)
plt.show()
# -----------------------------------------------------------------------------------------------


# Load Probability map conversion ---------------------------------------------------------------
index = 0
filter_size = 3 # 1 = 3x3, 2 = 5x5, 3 = 7x7
classify = True
map = grid_map_probability(index,filter_size,classify)
plt.imshow(map)
plt.show()
# -----------------------------------------------------------------------------------------------


# dummy pmap ------------------------------------------------------------------------------------
map = pmap()
plt.imshow(map)
plt.show()
# -----------------------------------------------------------------------------------------------


# view map in config space folder ---------------------------------------------------------------
map_list = glob.glob('./map/mapdata/config_space_data_2d/*.npy')
fig, axs = plt.subplots(2, 3)
axs[0, 0].imshow(np.load(map_list[0]))
axs[0, 0].set_title('Axis [0, 0]')
axs[0, 1].imshow(np.load(map_list[1]))
axs[0, 1].set_title('Axis [0, 1]')
axs[0, 2].imshow(np.load(map_list[2]))
axs[1, 0].set_title('Axis [1, 0]')
axs[1, 0].imshow(np.load(map_list[3]))
axs[1, 1].set_title('Axis [1, 1]')
axs[1, 1].imshow(np.load(map_list[4]))
axs[1, 1].set_title('Axis [1, 1]')
plt.show()
# -----------------------------------------------------------------------------------------------
