import numpy as np
import matplotlib.pyplot as plt
from load_map import grid_map_binary, grid_map_probability
from dummy_map import pmap

# View map -------------------------------------------------------------------------------------
# map = np.load("./map/mapdata/task_space/grid_113.npy").astype(np.uint8)
# print(map.shape)
# plt.imshow(map)
# plt.show()

# index = 0
# map = grid_map_binary(index)
# print(map)
# plt.imshow(map)
# plt.show()
# -----------------------------------------------------------------------------------------------



# Load Probability map conversion ---------------------------------------------------------------
index = 0
filter_size = 3 # 1 = 3x3, 2 = 5x5, 3 = 7x7
classify = True
map = grid_map_probability(index,filter_size,classify)
print(map)
plt.imshow(map)
plt.show()

map = 1 - map #np.transpose(1-map2)
print(map)
plt.imshow(map)
plt.show()
# -----------------------------------------------------------------------------------------------


# dummy pmap
map = pmap()
plt.imshow(map)
plt.show()


# conf2d = np.load('./map/mapdata/config_space_data_2d/path(leaf-x)-5.npy')
# print(conf2d.shape)
# plt.imshow(conf2d)
# plt.show()

# conf3d = np.load('./map/mapdata/config_space_data_3d/config3D.npy')
# plt.imshow(conf2d[0])
# plt.show()

# point_cloud_rgb = np.load('./map/mapdata/point_cloud/rgb(0.3).npy')
# print(point_cloud_rgb.shape)

# point_cloud_xyz = np.load('./map/mapdata/point_cloud/xyz(0.3).npy')
# print(point_cloud_xyz.shape)