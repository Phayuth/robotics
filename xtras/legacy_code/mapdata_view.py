import numpy as np
import glob
import matplotlib.pyplot as plt


# SECTION - view 2D map in config space folder
map_list = glob.glob('./datasave/config_space_data_2d/*.npy')
fig, axs = plt.subplots(1, len(map_list))
for i in range(len(map_list)):
    axs[i].imshow(np.load(map_list[i]))
    axs[i].set_title(f'ID : {i}')
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