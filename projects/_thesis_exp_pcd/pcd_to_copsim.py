import os
import sys

sys.path.append(str(os.path.abspath(os.getcwd())))
sys.path.append("/home/yuth/coppelia/programming/zmqRemoteApi/clients/python")

from zmqRemoteApi import RemoteAPIClient
import numpy as np
import open3d as o3d


# client = RemoteAPIClient()
# sim = client.getObject("sim")


pcd = o3d.io.read_point_cloud("/home/yuth/experiment/Reconstruction3D/scan7_used_in_thesis/multiway_registration_full.ply")
o3d.visualization.draw_geometries([pcd], window_name="og pcd")

# pcd = pcd.voxel_down_sample(voxel_size=0.005)
# pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
# o3d.visualization.draw_geometries([pcd], window_name="pcd down")


# def add_pcd_to_copsim(points, rgbs):
#     if isinstance(points, np.ndarray):
#         points = points.flatten().tolist()
#         rgbs = rgbs.astype(np.int32).flatten().tolist()
#     pointCloudHandles = sim.getObject("./PointCloud")
#     totalPointCnt = sim.insertPointsIntoPointCloud(pointCloudHandles, 0, points, rgbs)


# # points = np.asarray(pcd.points)
# # rgbs = np.asarray(pcd.colors) * 255
# # add_pcd_to_copsim(points, rgbs)

# # mesh method 1 BPA
# distance = pcd.compute_nearest_neighbor_distance()
# print(f"> distance: {distance}")
# avg_dist = np.mean(distance)
# print(f"> avg_dist.shape: {avg_dist.shape}")
# radius = 3 * avg_dist
# print(f"> radius: {radius}")

# bpa_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd, o3d.utility.DoubleVector([radius, radius * 2]))
# o3d.visualization.draw_geometries([bpa_mesh], window_name="BPA")


# if True:
#     o3d.io.write_triangle_mesh("bpa_mesh.ply", bpa_mesh)
