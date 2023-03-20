import numpy as np
import glob
import matplotlib.pyplot as plt

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

# plot config space in 3d #slow
map = np.load('./map/mapdata/config_space_data_3d/config3D.npy')
print("==>> map.shape: \n", map.shape)
ax = plt.figure().add_subplot(projection='3d')
ax.voxels(map,  edgecolor='k')
plt.show()
