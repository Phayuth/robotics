import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R


class HarvestPose:

    def __init__(self, pointArray) -> None:
        self.pcd = o3d.geometry.PointCloud()
        self.pcd.points = o3d.utility.Vector3dVector(pointArray)
        self.pcd.estimate_normals()
        self.pcd.normalize_normals()
        self.pcd.orient_normals_towards_camera_location()
        self.pcdDownSample = self.pcd.voxel_down_sample(voxel_size=0.01)  #0.01

        self.point = np.asarray(self.pcdDownSample.points).T
        self.normalPointOut = np.asarray(self.pcdDownSample.normals).T
        self.auxPoint, self.normalPointIntoCrop = self.aux_pose_offset(self.point, self.normalPointOut * -1, distanceOffset=0.1)  #(3, N)

    def get_point(self):
        return self.point, self.auxPoint, self.normalPointIntoCrop

    def vec_to_skew(self, vector):
        matrix = np.array([[0, -vector[2], vector[1]], [vector[2], 0, -vector[0]], [-vector[1], vector[0], 0]])
        return matrix

    def aux_pose_offset(self, point, normalPointIntoCrop, distanceOffset):
        if point.shape[0] != 3:
            point = point.T
            normalPointIntoCrop = normalPointIntoCrop.T
        auxPoint = distanceOffset * (-1 * normalPointIntoCrop) + point
        return auxPoint, normalPointIntoCrop

    def rotmat_align_z_axis(self, unitVector):
        zAxis = np.array([0, 0, 1])
        rotAxis = np.cross(zAxis, unitVector)
        rotAxis /= np.linalg.norm(rotAxis) + 0.0001  # to avoid devide by 0
        rotCos = np.dot(zAxis, unitVector)
        rotSin = np.sqrt(1 - rotCos**2)
        rotMat = np.eye(3) + rotSin * self.vec_to_skew(rotAxis) + (1-rotCos) * self.vec_to_skew(rotAxis) @ self.vec_to_skew(rotAxis)
        return rotMat

    def plot_quiver(self, ax):
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_xlim3d(-0.5, 0.5)
        ax.set_ylim3d(-0.5, 0.5)
        ax.set_zlim3d(-0.5, 0.5)
        ax.plot(self.point[0], self.point[1], self.point[2], "bo")
        ax.plot(self.auxPoint[0], self.auxPoint[1], self.auxPoint[2], "ro")
        ax.quiver(self.point[0], self.point[1], self.point[2], self.normalPointIntoCrop[0], self.normalPointIntoCrop[1], self.normalPointIntoCrop[2], length=0.01, normalize=True)
        ax.quiver(self.auxPoint[0], self.auxPoint[1], self.auxPoint[2], self.normalPointIntoCrop[0], self.normalPointIntoCrop[1], self.normalPointIntoCrop[2], length=0.01, normalize=True, color="orange")

    def plot_frame(self, ax, RM, origin=(0, 0, 0), scale=1):
        colors = ("#FF6666", "#005533", "#1199EE")
        loc = np.array([origin, origin])
        for i, (axis, c) in enumerate(zip((ax.xaxis, ax.yaxis, ax.zaxis), colors)):
            axlabel = axis.axis_name
            axis.set_label_text(axlabel)
            axis.label.set_color(c)
            axis.line.set_color(c)
            axis.set_tick_params(colors=c)
            line = np.zeros((2, 3))
            line[1, i] = scale
            line_rot = RM.apply(line)
            line_plot = line_rot + loc
            ax.plot(line_plot[:, 0], line_plot[:, 1], line_plot[:, 2], c)

    def plot_transformation(self, ax):
        # plot world origin frame
        RMworld = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        RMworldScipy = R.from_matrix(RMworld)
        self.plot_frame(ax, RMworldScipy, origin=(0, 0, 0))

        # plot all harvest and aux frame
        for i in range(point.shape[1]):
            RMHarvest = harvest.rotmat_align_z_axis(normalPointIntoCrop[:, i])
            RMHarvestScipy = R.from_matrix(RMHarvest)
            self.plot_frame(ax, RMHarvestScipy, origin=(point[0, i], point[1, i], point[2, i]), scale=0.01)
            self.plot_frame(ax, RMHarvestScipy, origin=(auxPoint[0, i], auxPoint[1, i], auxPoint[2, i]), scale=0.01)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    xyz = np.load("./datasave/grasp_poses/pose_array.npy")
    harvest = HarvestPose(xyz)
    point, auxPoint, normalPointIntoCrop = harvest.get_point()

    # plot with plt
    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_subplot(111, projection='3d')
    harvest.plot_quiver(ax)
    plt.show()

    fig = plt.figure()
    ax = fig.add_subplot(projection="3d", proj_type="ortho")
    harvest.plot_transformation(ax)
    plt.show()