import numpy as np
import matplotlib.pyplot as plt
import glob
from generate_map import grid_map_binary, grid_map_probability
from grid_map_sampling import Sampling

# View map
# map = np.load("./map/mapdata/task_space/grid_113.npy").astype(np.uint8)
# print(map.shape)
# plt.imshow(map)
# plt.show()

index = 0
map = grid_map_binary(index)
print(map)
plt.imshow(map)
plt.show()

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



# Grid Sampling
map1 = grid_map_probability(0, 0, False)
map2 = grid_map_probability(0, 3, False)
map3 = grid_map_probability(0, 3, True)

plt.figure(figsize=(10,10))
plt.axes().set_aspect('equal')
for i in range(1000):
    x = np.random.uniform(low = 0, high = map1.shape[0])
    y = np.random.uniform(low = 0, high = map1.shape[1])
    plt.scatter(x, y, c="red", s=2)
    if i%100 == 0:
        print(i)

plt.imshow(np.transpose(map1),cmap = "gray", interpolation = 'nearest')
plt.show()

plt.figure(figsize=(10,10))
plt.axes().set_aspect('equal')
for i in range(1000):
    x = Sampling(map1)
    plt.scatter(x[0], x[1], c="red", s=2)
    if i%100 == 0:
        print(i)

plt.imshow(np.transpose(map1),cmap = "gray", interpolation = 'nearest')
plt.show()

plt.figure(figsize=(10,10))
plt.axes().set_aspect('equal')
for i in range(1000):
    x = Sampling(map2)
    plt.scatter(x[0], x[1], c="red", s=2)
    if i%100 == 0:
        print(i)

plt.imshow(np.transpose(map2),cmap = "gray", interpolation = 'nearest')
plt.show()

plt.figure(figsize=(10,10))
plt.axes().set_aspect('equal')
for i in range(1000):
    x = Sampling(map3)
    plt.scatter(x[0], x[1], c="red", s=2)
    if i%100 == 0:
        print(i)

plt.imshow(np.transpose(map3),cmap = "gray", interpolation = 'nearest')
plt.show()