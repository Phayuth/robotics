import numpy as np


def rotation_matrix_align_z_axis(unit_vector):
    z_axis = np.array([0, 0, 1])  # Z-axis
    rotation_axis = np.cross(z_axis, unit_vector)
    rotation_axis /= np.linalg.norm(rotation_axis) + 0.0001  # to avoid devide by 0
    rotation_cos = np.dot(z_axis, unit_vector)
    rotation_sin = np.sqrt(1 - rotation_cos**2)
    rotation_matrix = np.eye(3) + rotation_sin * skew_symmetric_matrix(rotation_axis) + (1-rotation_cos) * skew_symmetric_matrix(rotation_axis) @ skew_symmetric_matrix(rotation_axis)
    return rotation_matrix


def skew_symmetric_matrix(vector):
    matrix = np.array([[0, -vector[2], vector[1]], [vector[2], 0, -vector[0]], [-vector[1], vector[0], 0]])
    return matrix


def auxilary_pose_offset(point, normalPointIntoCrop, distanceOffset):  # assume the normal vector direction is pointing into the crop when pass in
    if point.shape[0] != 3:
        point = point.T
        normalPointIntoCrop = normalPointIntoCrop.T

    auxPoint = distanceOffset * (-1 * normalPointIntoCrop) + point

    return auxPoint, normalPointIntoCrop


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import open3d as o3d
    from scipy.spatial.transform import Rotation as R

    xyz = np.load("./target_localization/point_cloud/pose_array.npy")
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.estimate_normals()
    pcd.normalize_normals()
    pcd.orient_normals_towards_camera_location()
    pcdDownSample = pcd.voxel_down_sample(voxel_size=0.01) #0.01

    point = np.asarray(pcdDownSample.points).T
    normalPointOut = np.asarray(pcdDownSample.normals).T
    auxPoint, normalPointIntoCrop = auxilary_pose_offset(point, normalPointOut * -1, distanceOffset=0.1)  #(3, N)

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
    ax.plot(auxPoint[0], auxPoint[1], auxPoint[2], "ro")
    ax.quiver(point[0], point[1], point[2], normalPointIntoCrop[0], normalPointIntoCrop[1], normalPointIntoCrop[2], length=0.01, normalize=True)
    ax.quiver(auxPoint[0], auxPoint[1], auxPoint[2], normalPointIntoCrop[0], normalPointIntoCrop[1], normalPointIntoCrop[2], length=0.01, normalize=True, color="orange")
    plt.show()

    def plot_rotated_axes(ax, r, name=None, offset=(0, 0, 0), scale=1):
        colors = ("#FF6666", "#005533", "#1199EE")  # Colorblind-safe RGB
        loc = np.array([offset, offset])
        for i, (axis, c) in enumerate(zip((ax.xaxis, ax.yaxis, ax.zaxis), colors)):
            axlabel = axis.axis_name
            axis.set_label_text(axlabel)
            axis.label.set_color(c)
            axis.line.set_color(c)
            axis.set_tick_params(colors=c)
            line = np.zeros((2, 3))
            line[1, i] = scale
            line_rot = r.apply(line)
            line_plot = line_rot + loc
            ax.plot(line_plot[:, 0], line_plot[:, 1], line_plot[:, 2], c)

    frameWorld = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    frameWorld = R.from_matrix(frameWorld)

    ax = plt.figure().add_subplot(projection="3d", proj_type="ortho")
    plot_rotated_axes(ax, frameWorld, name="r0", offset=(0, 0, 0))
    for i in range(point.shape[1]):
        rotation_matrix = rotation_matrix_align_z_axis(normalPointIntoCrop[:, i])
        newfff = R.from_matrix(rotation_matrix)
        plot_rotated_axes(ax, newfff, name="r0", offset=(point[0, i], point[1, i], point[2, i]), scale=0.01)
        plot_rotated_axes(ax, newfff, name="r0", offset=(auxPoint[0, i], auxPoint[1, i], auxPoint[2, i]), scale=0.01)
    ax.set(xlim=(-1.25, 1.25), ylim=(-1.25, 1.25), zlim=(-1.25, 1.25))
    ax.set(xticks=[-1, 0, 1], yticks=[-1, 0, 1], zticks=[-1, 0, 1])
    ax.set_aspect("equal", adjustable="box")
    ax.figure.set_size_inches(6, 5)
    plt.tight_layout()
    plt.show()
