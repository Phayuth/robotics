import os
import sys
wd = os.path.abspath(os.getcwd())
sys.path.append(str(wd))

import numpy as np
import open3d as o3d
from spatial_geometry.spatial_transformation import RigidBodyTransformation as rbt


class NormalVectorParallelGraspPose:
    """
    [Summary] : Given Multiple grasp point (x,y,z) in shape of (n,3) on object in form of segmented Point Cloud,
    the method produces Grasp and Pregrasp Pose of object in form of Transformation Matrix.

    [Method] :

    - Given Point Cloud is downsampled based on voxel size and its noise is removed using radius outlier
    - Pregrasp Pose is offseted from Grasp Pose backwards in a fixed distance(m).

    Two option on orientation can be chosen :
    - (fixed=True), all z-axis pointing out in one direction.
    - (fixed=False), each z-axis pointing parallel of normal vector of point cloud.

    """
    def __init__(self, pointArray, fixedOrientation=True, distanceOffset=0.1, downSampleVoxelSize=0.01) -> None:
        self.pcd = o3d.geometry.PointCloud()
        self.pcd.points = o3d.utility.Vector3dVector(pointArray)
        self.pcd.estimate_normals()
        self.pcd.normalize_normals()
        self.pcd.orient_normals_towards_camera_location()
        self.pcd, _ = self.pcd.remove_radius_outlier(nb_points=16, radius=0.05)
        self.pcdDownSample = self.pcd.voxel_down_sample(voxel_size=downSampleVoxelSize)

        self.graspPoint = np.asarray(self.pcdDownSample.points).T

        if fixedOrientation is True:
            self.normalPointOut = np.zeros((3,self.graspPoint.shape[1]))
            self.normalPointOut[0] -= 1
        else:
            self.normalPointOut = np.asarray(self.pcdDownSample.normals).T

        self.preGraspPoint, self.normalPointIntoCrop = self.pregrasp_point_offset(self.graspPoint, self.normalPointOut * -1, distanceOffset=distanceOffset) # (3, N)

    def get_point(self):
        return self.graspPoint, self.preGraspPoint, self.normalPointIntoCrop

    def get_grasp_and_pregrasp_poses(self):
        grasp = [rbt.conv_rotmat_and_t_to_h(self.rotmat_align_z_axis(self.normalPointIntoCrop[:, i]), self.graspPoint[:, i]) for i in range(self.graspPoint.shape[1])]
        preGrasp = [rbt.conv_rotmat_and_t_to_h(self.rotmat_align_z_axis(self.normalPointIntoCrop[:, i]), self.preGraspPoint[:, i]) for i in range(self.graspPoint.shape[1])]
        return grasp, preGrasp

    def pregrasp_point_offset(self, point, normalPointIntoCrop, distanceOffset):
        if point.shape[0] != 3:
            point = point.T
            normalPointIntoCrop = normalPointIntoCrop.T
        preGraspPoint = distanceOffset * (-1 * normalPointIntoCrop) + point
        return preGraspPoint, normalPointIntoCrop

    def rotmat_align_z_axis(self, unitVector):
        zAxis = np.array([0, 0, 1])
        rotAxis = np.cross(zAxis, unitVector)
        rotAxis /= np.linalg.norm(rotAxis) + 0.0000000001  # to avoid devide by 0
        rotCos = np.dot(zAxis, unitVector)
        rotSin = np.sqrt(1 - rotCos**2)
        rotMat = np.eye(3) + rotSin * rbt.vec_to_skew(rotAxis) + (1-rotCos) * rbt.vec_to_skew(rotAxis) @ rbt.vec_to_skew(rotAxis)
        return rotMat


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from pytransform3d.transformations import plot_transform
    from pytransform3d.plot_utils import make_3d_axis

    xyz = np.load("./datasave/grasp_poses/pose_array.npy")
    harvest = NormalVectorParallelGraspPose(xyz, fixedOrientation=False)
    grasp, preGrasp = harvest.get_grasp_and_pregrasp_poses()

    # plot with pytransform3d plotter
    ax = make_3d_axis(ax_s=1, unit="m", n_ticks=6)
    plot_transform(ax=ax)
    for i in range(len(grasp)):
        plot_transform(ax, A2B=grasp[i], s=0.01)
        plot_transform(ax, A2B=preGrasp[i], s=0.01)
    plt.tight_layout()
    plt.show()