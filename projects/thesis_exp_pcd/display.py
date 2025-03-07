import os
import sys

sys.path.append(str(os.path.abspath(os.getcwd())))


import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
from spatial_geometry.spatial_transformation import RigidBodyTransformation as rbt
import pytransform3d.camera as pc
import pytransform3d.transformations as pt
from merge_harvest import transform_pcd, crop_pcd

# def tcp_to_h(tcp):
#     R, _ = cv2.Rodrigues(np.array(tcp[3:6]))
#     H = np.eye(4) @ rbt.ht(tcp[0], tcp[1], tcp[2])
#     H[:3, :3] = R
#     return H


# HcamleftTotool0 = rbt.ht(0.2, 0.2, 0.2)@ rbt.hrx(np.pi / 2) @ rbt.hrz(np.pi/2)

# # plot
# ax = pt.plot_transform(name="tool0", s=0.3)
# pt.plot_transform(ax, HcamleftTotool0, s=0.1, name="CAML")
# plt.show()

pcd = o3d.io.read_point_cloud("/home/yuth/meshscan/fused_point_cloud_wrt_first_tcp.ply")
pcd = crop_pcd(pcd)
o3d.visualization.draw([pcd])


pcd = o3d.io.read_point_cloud("/home/yuth/scan7/multiway_registration_full.ply")
o3d.visualization.draw([pcd])


xyz = np.load("./datasave/grasp_poses/pose_array.npy")
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(xyz)
pcd.estimate_normals()
pcd.normalize_normals()
pcd.orient_normals_towards_camera_location()
pcd, _ = pcd.remove_radius_outlier(nb_points=16, radius=0.05)
pcd.paint_uniform_color([1, 0, 0])
o3d.visualization.draw_geometries([pcd])
