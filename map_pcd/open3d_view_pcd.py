import open3d as o3d
import numpy as np

# View PC
ply_path = 'map_pcd/ply/testply.ply'
pcd = o3d.io.read_point_cloud(ply_path)
print(pcd)
print(np.asarray(pcd.points).shape)
o3d.visualization.draw_geometries([pcd])


# Create AABB and BB
aabb = pcd.get_axis_aligned_bounding_box()
print(aabb.get_center())
aabb.color = (1, 0, 0)
obb = pcd.get_oriented_bounding_box()
print(obb.get_center())
obb.color = (0, 1, 0)
o3d.visualization.draw_geometries([pcd, aabb, obb])



# Calculate Cluster DBSCAN clustering
with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
    labels = np.array(pcd.cluster_dbscan(eps=0.02, min_points=10, print_progress=True))

max_label = labels.max()
print(f"point cloud has {max_label + 1} clusters")