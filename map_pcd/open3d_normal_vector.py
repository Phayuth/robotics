import os
import sys
wd = os.path.abspath(os.getcwd())
sys.path.append(str(wd))
sys.path.append("/home/yuth/coppelia/programming/zmqRemoteApi/clients/python")

import matplotlib.pyplot as plt
import open3d as o3d
import numpy as np

mesh = o3d.io.read_triangle_mesh("map_pcd/stl/model_pkin.stl")
# mesh = o3d.io.read_triangle_mesh("map_pcd/stl/model_ccb.stl")
pointcloud = mesh.sample_points_poisson_disk(200)

# you can plot and check
# o3d.visualization.draw_geometries([pointcloud])

scale = 0.01
normal = -1*np.asarray(pointcloud.normals).T # unit vector (*-1 for reverse direction)
point = scale*np.asarray(pointcloud.points).T
print(f"==>> point.shape: \n{point.shape}")


fig = plt.figure(figsize=(4, 4))
ax = fig.add_subplot(111, projection='3d')

# set label name
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

ax.quiver(point[0], point[1], point[2], normal[0], normal[1], normal[2], length=scale*4, normalize=True)
plt.show()


def create_new_coordinate_system(unit_vector):
    # Step 1: Generate an arbitrary vector
    arbitrary_vector = np.array([1.0, 0.0, 0.0])
    
    # Step 2: Calculate the first orthogonal vector
    orthogonal_vector_1 = np.cross(unit_vector, arbitrary_vector)
    orthogonal_vector_1 /= np.linalg.norm(orthogonal_vector_1)
    
    # Step 3: Calculate the second orthogonal vector
    orthogonal_vector_2 = np.cross(unit_vector, orthogonal_vector_1)
    orthogonal_vector_2 /= np.linalg.norm(orthogonal_vector_2)
    
    # Step 4: Create the rotation matrix
    rotation_matrix = np.column_stack((orthogonal_vector_2, orthogonal_vector_1, unit_vector))
    
    return rotation_matrix

def rotation_matrix_to_euler_angles(rotation_matrix):
    sy = np.sqrt(rotation_matrix[0, 0] * rotation_matrix[0, 0] + rotation_matrix[1, 0] * rotation_matrix[1, 0])
    singular = sy < 1e-6

    if not singular:
        alpha = np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
        beta = np.arctan2(-rotation_matrix[2, 0], sy)
        gamma = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
    else:
        alpha = np.arctan2(-rotation_matrix[1, 2], rotation_matrix[1, 1])
        beta = np.arctan2(-rotation_matrix[2, 0], sy)
        gamma = 0

    return alpha, beta, gamma


# test in coppeliasim
from zmqRemoteApi import RemoteAPIClient
client = RemoteAPIClient()
sim = client.getObject('sim')

# Create a few dummies and set their positions:
handles = [sim.createDummy(0.01, 12 * [0]) for _ in range(point.shape[1])]
for i, h in enumerate(handles):
    sim.setObjectPosition(h, -1, [point[0,i], point[1,i], point[2,i]])
    RM = create_new_coordinate_system(normal[:,i])
    a, b, g = rotation_matrix_to_euler_angles(RM)
    sim.setObjectOrientation(h, -1, [a,b,g])
