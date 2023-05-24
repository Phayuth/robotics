import os
import sys

wd = os.path.abspath(os.getcwd())
sys.path.append(str(wd))

import numpy as np
import matplotlib.pyplot as plt
from planner_util.coord_transform import sph2cat


# SECTION - target define pose
x_targ = 0.5
y_targ = 0.5
z_targ = 0.5
theta_range = np.linspace(0, np.pi, 10)
phi_range = np.linspace(np.pi / 2, 3 * np.pi / 2, 10)

index = 6
x, y, z = sph2cat(3, theta_range[index], phi_range[index])

r = np.linalg.norm([x, y, z])
theta = np.arccos(z / r)
phi = np.sign(y) * np.arccos(x / np.linalg.norm([x, y]))


# SECTION - create inner and outer pose of vector
cand_outer = []
for th in theta_range:
    for ph in phi_range:
        x, y, z = sph2cat(3, th, ph)
        cand_outer.append([x, y, z])
cand_outer = np.array(cand_outer)
cand_inner = []
for th in theta_range:
    for ph in phi_range:
        x, y, z = sph2cat(2, th, ph)
        cand_inner.append([x, y, z])
cand_inner = np.array(cand_inner)


# SECTION - plot task space
ax = plt.axes(projection='3d')
ax.plot3D([0, 1], [0, 0], [0, 0], 'red', linewidth=4)
ax.plot3D([0, 0], [0, 1], [0, 0], 'purple', linewidth=4)
ax.plot3D([0, 0], [0, 0], [0, 1], 'gray', linewidth=4)

for i in range(len(cand_outer)):
    ax.plot3D([cand_inner[i, 0], cand_outer[i, 0]], [cand_inner[i, 1], cand_outer[i, 1]], [cand_inner[i, 2], cand_outer[i, 2]], 'blue', linewidth=2)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

plt.show()