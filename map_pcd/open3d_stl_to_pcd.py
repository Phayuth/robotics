import os
import sys
wd = os.path.abspath(os.getcwd())
sys.path.append(str(wd))

import open3d as o3d
import numpy as np

print("Testing IO for meshes ...")
mesh = o3d.io.read_triangle_mesh("map_pcd/stl/model_pkin.stl")
print(mesh)

# convert stl to point cloud / https://github.com/isl-org/Open3D/issues/867
pointcloud = mesh.sample_points_poisson_disk(100)

# you can plot and check
o3d.visualization.draw_geometries([mesh])
o3d.visualization.draw_geometries([pointcloud])

normal = np.asarray(pointcloud.normals)
print(f"==>> normal: \n{normal}")

a = normal[0,:]
norm = np.linalg.norm(a)
print(f"==>> norm: \n{norm}")
# point = np.asarray(pointcloud.points)
# print(f"==>> point: \n{point}")

# print("Try to render a mesh with normals (exist: " +
#         str(mesh.has_vertex_normals()) + ") and colors (exist: " +
#         str(mesh.has_vertex_colors()) + ")")
# o3d.visualization.draw_geometries([mesh])
# print("A mesh with no normals and no colors does not seem good.")


	

# print("Computing normal and rendering it.")
# mesh.compute_vertex_normals()
# mesh.create_arrow()
# print(np.asarray(mesh.triangle_normals))
# o3d.visualization.draw_geometries([mesh])

# print("Downsample the point cloud with a voxel of 0.05")
# downpcd = o3d.geometry.voxel_down_sample(pointcloud, voxel_size=0.05)
# o3d.visualization.draw_geometries([downpcd])


# print("Recompute the normal of the downsampled point cloud")
# o3d.geometry.estimate_normals(downpcd,search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1,max_nn=30))
# o3d.visualization.draw_geometries([downpcd])


# print("We make a partial mesh of only the first half triangles.")
# mesh1 = copy.deepcopy(mesh)
# mesh1.triangles = o3d.utility.Vector3iVector(
#     np.asarray(mesh1.triangles)[:len(mesh1.triangles) // 2, :])
# mesh1.triangle_normals = o3d.utility.Vector3dVector(
#     np.asarray(mesh1.triangle_normals)[:len(mesh1.triangle_normals) //
#                                         2, :])
# print(mesh1.triangles)
# o3d.visualization.draw_geometries([mesh1])

	

# print("Painting the mesh")
# mesh1.paint_uniform_color([1, 0.706, 0])
# o3d.visualization.draw_geometries([mesh1])

