import os
import sys
wd = os.path.abspath(os.getcwd())
sys.path.append(str(wd))
from planner.research_potfield.potential_field import calc_potential_field, potential_field_planning
from map.load_map import grid_map_binary
import matplotlib.pyplot as plt
import numpy as np

# Load map
index = 2
map = grid_map_binary(index)  # 1 is free, 0 is obstacle
print("Map size", len(map[0]), len(map[1]))
# plt.imshow(map)
# plt.show()

# set start and goal pose
sx = 23  # start x
sy = 2  # start y
gx = 20  # goal x
gy = 27  # goal y
robot_radius = 2  # robot radius

# # Create a potential map
potential_map, minx, miny, maxx, maxy = calc_potential_field(map, gx, gy, robot_radius, sx, sy)
# # print(minx, miny, maxx, maxy)
plt.imshow(potential_map,origin='lower')
plt.show()




# # have to transpose map cause of coordinate system getting weird
# potential_map, rx, ry = potential_field_planning(map, sx, sy, gx, gy, robot_radius)
# print(rx)
# print(ry)


# plt.plot(rx,ry)
# # plt.imshow(potential_map, origin='lower')
# plt.imshow(potential_map)
# plt.show()

# View map in 3d
z = np.array(potential_map)
x = np.arange(len(z[0]))
y = np.arange(len(z[1]))
x, y = np.meshgrid(x, y)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(x, y, z)
plt.title('Art. Pot. Map')
plt.show()








# Search of the min and max potential value in map
# pmap_np = np.array(pmap)
# minv = min(pmap_np.flatten())
# maxv = max(pmap_np.flatten())
# # print(minv, maxv)

# mapping potential value to range 0 , 255
# pmap_mapv = np.zeros_like(pmap_np)

# for i in range(len(pmap_mapv[0])):
#     for j in range(len(pmap_mapv[1])):
#         pmap_mapv[i][j] = map_val(pmap_np[i][j],minv,maxv,0,255)

# minv = min(pmap_mapv.flatten())
# maxv = max(pmap_mapv.flatten())

# # print(minv, maxv)

# plt.imshow(pmap_mapv, cmap='viridis')
# plt.colorbar()
# plt.show()
