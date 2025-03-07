import cv2
import numpy as np
import matplotlib.pyplot as plt
import pytransform3d.camera as pc
import pytransform3d.transformations as pt

from pytransform3d.transformations import plot_transform
from pytransform3d.plot_utils import make_3d_axis
from icecream import ic
import os
import sys

wd = os.path.abspath(os.getcwd())
sys.path.append(str(wd))


import numpy as np
from spatial_geometry.spatial_transformation import RigidBodyTransformation as rbt


np.set_printoptions(suppress=True, precision=4)

# def stereo_old():
#     intrinsic = np.array([1020 / 0.194, 0, 768, 0, 1020 / 0.194, 768, 0, 0, 1]).reshape(3, 3)
#     RT1 = np.concatenate([np.eye(3), [[0], [0], [0]]], axis=-1)  # RT matrix for C1 is identity.
#     p1 = intrinsic @ RT1

#     HpToC1 = np.array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 605, 0, 0, 0, 1]).reshape(4, 4)
#     HpToC2 = np.array([1, 0, 0, 0, 0, 0, 1, 0, 0, -1, 0, 500, 0, 0, 0, 1]).reshape(4, 4)
#     HC2ToC1 = HpToC1 @ rbt.hinverse(HpToC2)
#     RT2 = HC2ToC1[0:3, :]  # RT matrix for C2 is the R and T obtained from stereo calibration.
#     p2 = intrinsic @ RT2

#     ax = make_3d_axis(ax_s=100, unit="m", n_ticks=6)
#     plot_transform(ax=ax, s=100, name="cam1")
#     plot_transform(ax=ax, A2B=HC2ToC1, s=100, name="cam2tocam1")
#     plot_transform(ax=ax, A2B=HpToC1, s=100, name="p to cam1")
#     plot_transform(ax=ax, A2B=HpToC2, s=100, name="p to cam2")
#     plt.tight_layout()
#     plt.show()

#     return p1, p2


# def stereo():
#     intrinsic = np.array([1020 / 0.194, 0, 768, 0, 1020 / 0.194, 768, 0, 0, 1]).reshape(3, 3)

#     Hp1ToC1 = np.array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 500, 0, 0, 0, 1]).reshape(4, 4)

#     # Hp2Top1 = np.array([1, 0, 0, 0, 0, 0, 1, 0, 0, -1, 0, 700, 0, 0, 0, 1]).reshape(4, 4)
#     Hp2ToC2 = np.array([1, 0, 0, 0, 0, 0, 1, 0, 0, -1, 0, 700, 0, 0, 0, 1]).reshape(4, 4)

#     Hp2Top1 = rbt.hinverse(Hp1ToC1) @ Hp2ToC2
#     HC2Top2 = rbt.hinverse(Hp1ToC1)

#     Hp2ToC1 = Hp1ToC1 @ Hp2Top1
#     HC2ToC1 = Hp2ToC1 @ HC2Top2

#     ax = make_3d_axis(ax_s=100, unit="m", n_ticks=6)
#     plot_transform(ax=ax, s=100, name="cam1")
#     plot_transform(ax=ax, A2B=Hp1ToC1, s=100, name="p1")
#     plot_transform(ax=ax, A2B=Hp2ToC1, s=100, name="p2")
#     plot_transform(ax=ax, A2B=HC2ToC1, s=100, name="cam2")
#     plt.tight_layout()
#     plt.show()

#     RT1 = np.concatenate([np.eye(3), [[0], [0], [0]]], axis=-1)  # RT matrix for C1 is identity.
#     p1 = intrinsic @ RT1

#     RT2 = HC2ToC1[0:3, :]  # RT matrix for C2 is the R and T obtained from stereo calibration.
#     p2 = intrinsic @ RT2

#     return p1, p2


def stereo_new():
    fl = 1020 / 0.194
    # fl = 593
    K = np.array([[fl, 0, 768], [0, fl, 768], [0, 0, 1]])

    Mint = np.hstack((K, np.zeros((3, 1))))
    ic(Mint)

    HC2ToC1 = rbt.ht(600, 0, 500) @ rbt.hry(-np.pi / 2)
    ic(HC2ToC1)

    HC1ToC2 = rbt.hinverse(HC2ToC1)

    ax = make_3d_axis(ax_s=100, unit="mm", n_ticks=10)
    plot_transform(ax=ax, s=100, name="cam1")
    # plot_transform(ax=ax, A2B=HpToC1, s=100, name="p1")
    plot_transform(ax=ax, A2B=HC2ToC1, s=100, name="cam2")
    plt.tight_layout()
    plt.show()

    p1 = Mint
    ic(p1)

    p2 = Mint @ HC1ToC2
    ic(p2)

    return p1, p2


def triangulate(p1, p2, point1, point2):
    point1 = point1.astype(np.float64)
    point2 = point2.astype(np.float64)

    point3d = cv2.triangulatePoints(p1, p2, point1, point2, None)
    point3d /= point3d[3, :]
    return point3d


def DLT(P1, P2, point1, point2):

    A = [point1[1] * P1[2, :] - P1[1, :], P1[0, :] - point1[0] * P1[2, :], point2[1] * P2[2, :] - P2[1, :], P2[0, :] - point2[0] * P2[2, :]]
    A = np.array(A).reshape((4, 4))

    B = A.transpose() @ A
    from scipy import linalg

    U, s, Vh = linalg.svd(B, full_matrices=False)

    print("Triangulated point: ")
    print(Vh[3, 0:3] / Vh[3, 3])
    return Vh[3, 0:3] / Vh[3, 3]


# print(cv2.__version__)
# p1, p2 = stereo_old()
# p1, p2 = stereo()
p1, p2 = stereo_new()
print(f"> p1.shape: {p1.shape}")
print(f"> p2.shape: {p2.shape}")


# --------------
uv1 = np.array([360.0, 853.0])
uv2 = np.array([693.0, 588.0])

p3d = triangulate(p1, p2, uv1, uv2)
print(f"> p3d: {p3d}")

# p3d = DLT(p1, p2, uv1, uv2)
# print(f"> p3d: {p3d}")


# find uv from 3d
# point3d = np.array([-82, 58, 500.0, 1]).reshape(4, 1)

uv1 = p1 @ p3d
uv1 = uv1[:2, :] / uv1[2, :]
print(f"> uv1: {uv1}")

uv2 = p2 @ p3d
uv2 = uv2[:2, :] / uv2[2, :]
print(f"> uv2: {uv2}")

# p3d = triangulate(p1, p2, uv1, uv2)
# print(f"> p3d: {p3d}")
