import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt


def example():
    # load stl
    mesh = o3d.io.read_triangle_mesh('./datasave/stl/model_pkin.stl')
    pcd = mesh.sample_points_poisson_disk(500)

    # load ply
    pcd = o3d.io.read_point_cloud('./datasave/ply/testply.ply')

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

    # paint and view
    pcd.paint_uniform_color([1, 0, 0])
    pcdDownSample.paint_uniform_color([0, 0, 1])
    o3d.visualization.draw_geometries([pcd, pcdDownSample])


if __name__=="__main__":
    # example()
    process_pose_array()