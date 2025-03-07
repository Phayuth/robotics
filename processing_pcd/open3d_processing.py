import os
import sys

sys.path.append(str(os.path.abspath(os.getcwd())))


import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from spatial_geometry.spatial_transformation import RigidBodyTransformation as rbt


def example():
    # load stl
    mesh = o3d.io.read_triangle_mesh("./datasave/stl/model_pkin.stl")
    pcd = mesh.sample_points_poisson_disk(500)

    # load ply
    pcd = o3d.io.read_point_cloud("./datasave/ply/testply.ply")

    # load numpy
    xyz = np.load("./datasave/grasp_poses/pose_array.npy")
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.estimate_normals()
    pcd.normalize_normals()
    pcd.orient_normals_towards_camera_location()

    # downsample
    pcdDownSample = pcd.uniform_down_sample(20)
    pcdDownSample = pcd.random_down_sample(sampling_ratio=1)
    pcdDownSample = pcd.voxel_down_sample(voxel_size=0.01)

    # extract point
    normal = np.asarray(pcdDownSample.normals).T
    point = np.asarray(pcdDownSample.points).T

    # plot
    o3d.visualization.draw_geometries([pcdDownSample, mesh])

    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_xlim3d(-0.5, 0.5)
    ax.set_ylim3d(-0.5, 0.5)
    ax.set_zlim3d(-0.5, 0.5)
    ax.plot(point[0], point[1], point[2], "bo")
    ax.quiver(point[0], point[1], point[2], normal[0], normal[1], normal[2], length=0.01, normalize=True)
    plt.show()


def process_pose_array():
    xyz = np.load("./datasave/grasp_poses/pose_array.npy")
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.estimate_normals()
    pcd.normalize_normals()
    pcd.orient_normals_towards_camera_location()

    # remove noise
    # cl, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    pcd, ind = pcd.remove_radius_outlier(nb_points=16, radius=0.05)

    # downsample
    pcdDownSample = pcd.voxel_down_sample(voxel_size=0.005)

    # arrow
    arw1 = o3d.geometry.TriangleMesh.create_arrow(cylinder_radius=0.002, cylinder_height=0.02, cone_radius=0.003, cone_height=0.01)
    arw2 = o3d.geometry.TriangleMesh.create_arrow(cylinder_radius=0.002, cylinder_height=0.02, cone_radius=0.003, cone_height=0.01)
    arw3 = o3d.geometry.TriangleMesh.create_arrow(cylinder_radius=0.002, cylinder_height=0.02, cone_radius=0.003, cone_height=0.01)
    H = rbt.hry(np.pi/2)
    H = H @ rbt.ht(0,0.01,0.35)
    arw1.transform(H)
    arw1.paint_uniform_color([0, 0, 1])
    arw1.compute_vertex_normals()

    H = rbt.hry(np.pi/2+0.1)
    H = H @ rbt.ht(-0.05,0.01,0.35)
    arw2.transform(H)
    arw2.paint_uniform_color([0, 0, 1])
    arw2.compute_vertex_normals()

    H = rbt.hry(np.pi/2-0.1)
    H = H @ rbt.ht(0.05,0.01,0.35)
    arw3.transform(H)
    arw3.paint_uniform_color([0, 0, 1])
    arw3.compute_vertex_normals()

    # paint and view
    pcd.paint_uniform_color([1, 0, 0])
    pcdDownSample.paint_uniform_color([0, 0, 1])
    o3d.visualization.draw_geometries([pcd, arw1, arw2, arw3])


def pcd_load_zed():
    xyz = np.load("./datasave/grasp_poses/pointcloud.npy")
    # xyz = xyz*0.0000001
    print(f"> xyz.shape: {xyz.shape}")
    print(f"> xyz: {xyz}")
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.paint_uniform_color([1, 0.706, 0])
    pcdDownSample = pcd.uniform_down_sample(30)
    print(f"> pcdDownSample: {pcdDownSample}")

    o3d.visualization.draw_geometries([pcdDownSample])

    # normal = np.asarray(pcdDownSample.normals).T
    # point = np.asarray(pcdDownSample.points).T
    # print(f"> point.shape: {point.shape}")

    # fig = plt.figure(figsize=(4, 4))
    # ax = fig.add_subplot(111, projection="3d")
    # ax.set_xlabel("X")
    # ax.set_ylabel("Y")
    # ax.set_zlabel("Z")
    # ax.set_xlim3d(-0.5, 0.5)
    # ax.set_ylim3d(-0.5, 0.5)
    # ax.set_zlim3d(-0.5, 0.5)
    # ax.plot(point[0], point[1], point[2], "bo")
    # plt.show()


def pcdrgb_load_zed():
    xyzc = np.load("./datasave/grasp_poses/pointcloudrgb.npy")
    xyz = xyzc[:, 0:3]
    # color = xyzc[:,3,np.newaxis]

    nan_mask = np.isnan(xyz)
    nan_rows = np.any(nan_mask, axis=1)
    xyzfilter = xyz[~nan_rows]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyzfilter)
    # pcd.colors = o3d.utility.Vector3dVector(xyzc[:,3])
    o3d.visualization.draw_geometries([pcd])

    # o3d.io.write_point_cloud("output.ply", pcd)

    point = np.asarray(pcd.points).T

    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_xlim3d(-0.5, 0.5)
    ax.set_ylim3d(-0.5, 0.5)
    ax.set_zlim3d(-0.5, 0.5)
    ax.plot(point[0], point[1], point[2], "bo")
    plt.show()


if __name__ == "__main__":
    # example()
    # process_pose_array()
    # pcd_load_zed()
    pcdrgb_load_zed()
