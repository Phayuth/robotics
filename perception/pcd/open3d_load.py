"""
http://www.open3d.org/docs/0.9.0/python_api/open3d.geometry.PointCloud.html
http://www.open3d.org/docs/release/tutorial/geometry/working_with_numpy.html?highlight=numpy

"""
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

# Pointcloud from stl https://github.com/isl-org/Open3D/issues/867
# mesh = o3d.io.read_triangle_mesh("map_pcd/stl/model_pkin.stl")
# pc = mesh.sample_points_poisson_disk(500)

# Pointcloud from numpy
# xyz = np.load("./map_pcd/pose_array.npy")
# pcd = o3d.geometry.PointCloud()
# pcd.points = o3d.utility.Vector3dVector(xyz)
# pcd.estimate_normals()
# pcd.normalize_normals()
# pcd.orient_normals_towards_camera_location()

# Pointcloud from PLY
ply_path = 'map_pcd/ply/testply.ply'
pcd = o3d.io.read_point_cloud(ply_path)

# Acessing info
point = np.asarray(pcd.points)  # point
normal = np.asarray(pcd.normals)  # normal vector

# Visual
# o3d.visualization.draw_geometries([mesh])
o3d.visualization.draw_geometries([pcd])

# Plot with plt
fig = plt.figure(figsize=(4, 4))
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_xlim3d(-0.5, 0.5)
ax.set_ylim3d(-0.5, 0.5)
ax.set_zlim3d(-0.5, 0.5)
point = point.T
normal = -1 * normal.T
ax.plot(point[0], point[1], point[2], "bo")
ax.quiver(point[0], point[1], point[2], normal[0], normal[1], normal[2], length=0.01, normalize=True)
plt.show()