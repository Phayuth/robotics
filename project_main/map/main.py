import numpy as np
import matplotlib.pyplot as plt
import generate_map
import glob

# map = np.load("./map/mapdata/task_space/grid_113.npy").astype(np.uint8)
# print(map.shape)
# plt.imshow(map)
# plt.show()

# index = 0
# map = generate_map.grid_map_binary(index)
# print(map)
# plt.imshow(map)
# plt.show()

# filter_size = 3 # 1 = 3x3, 2 = 5x5, 3 = 7x7
# classify = True
# map = generate_map.grid_map_probability(index,filter_size,classify)
# print(map)
# plt.imshow(map)
# plt.show()

# map = 1 - map #np.transpose(1-map2)
# print(map)
# plt.imshow(map)
# plt.show()

# conf = np.load('./map/mapdata/config_space_data_2d/path(leaf-x)-5.npy')
# print(conf.shape)
# plt.imshow(conf)
# plt.show()

# point_cloud_rgb = np.load('./map/mapdata/point_cloud/rgb(0.3).npy')
# print(point_cloud_rgb.shape)

# point_cloud_xyz = np.load('./map/mapdata/point_cloud/xyz(0.3).npy')
# print(point_cloud_xyz.shape)

conf1 = np.load('./map/mapdata/config_space_data_3d/2/config3D.npy')
conf2 = np.load('./map/mapdata/config_space_data_3d/2/config3D(73).npy')
print(np.array_equal(conf1,conf2))