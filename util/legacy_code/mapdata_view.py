import numpy as np
import glob
import matplotlib.pyplot as plt


# SECTION - view 2D map in config space folder
map_list = glob.glob('./datasave/config_space_data_2d/*.npy')
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


# SECTION - point cloud
map = np.load('./datasave/point_cloud/xyz(0.3).npy')
print(f"==>> map.shape: {map.shape}")
x_coord = map[:,0]
y_coord = map[:,1]
z_coord = map[:,2]

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.plot(x_coord, y_coord, z_coord, linewidth=0, marker='o', markerfacecolor='darkcyan', markersize=1.5)
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
plt.show()