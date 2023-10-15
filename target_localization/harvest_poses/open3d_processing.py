import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

# load
xyz = np.load("./target_localization/point_cloud/pose_array.npy")
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(xyz)
pcd.estimate_normals()
pcd.normalize_normals()
pcd.orient_normals_towards_camera_location()

# pcdDownSample = pcd.uniform_down_sample(20)
# pcdDownSample = pcd.random_down_sample(sampling_ratio=1)
pcdDownSample = pcd.voxel_down_sample(voxel_size=0.01)

# plot with openviz
o3d.visualization.draw_geometries([pcdDownSample])

normal = -1 * np.asarray(pcdDownSample.normals).T
point = np.asarray(pcdDownSample.points).T

# plot with plt
fig = plt.figure(figsize=(4, 4))
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_xlim3d(-0.5, 0.5)
ax.set_ylim3d(-0.5, 0.5)
ax.set_zlim3d(-0.5, 0.5)
ax.plot(point[0], point[1], point[2], "bo")
ax.quiver(point[0], point[1], point[2], normal[0], normal[1], normal[2], length=0.01, normalize=True)
plt.show()
