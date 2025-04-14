import cv2
import numpy as np

np.set_printoptions(suppress=True, linewidth=200)
np.random.seed(10)

import matplotlib.pyplot as plt
from pytransform3d.transformations import plot_transform
from pytransform3d.plot_utils import make_3d_axis


def collinear_eqn():
    fx = 200; fy = 200; x0 = 100; y0 = 100; k1 = 0.2; k2 = 0.2; k3 = 0.2; p1 = 0.2; p2 = 0.2
    r11 = 1; r12 = 0; r13 = 0; r21 = 0; r22 = 1; r23 = 0; r31 = 0; r32 = 0; r33 = 1
    Xs = 1
    Ys = 1
    Zs = 1
    optv = [fx, fy, x0, y0, k1, k2, k3, p1, p2, r11, r12, r13, r21, r22, r23, r31, r32, r33, Xs, Ys, Zs]
    X, Y, Z = [0, 0, 0]
    u = x0 - fx * (r11 * (X - Xs) + r12 * (Y - Ys) + r13 * (Z - Zs)) / (r31 * (X - Xs) + r32 * (Y - Ys) + r33 * (Z - Zs))
    v = y0 - fy * (r21 * (X - Xs) + r22 * (Y - Ys) + r23 * (Z - Zs)) / (r31 * (X - Xs) + r32 * (Y - Ys) + r33 * (Z - Zs))


def plot_stuff(Mext, points3d):
    HCamToWorld = np.linalg.inv(Mext)

    ax = make_3d_axis(ax_s=1, unit="m", n_ticks=6)
    plot_transform(ax=ax, name="world")
    plot_transform(ax=ax, A2B=HCamToWorld, name="camera")
    ax.plot(points3d[0], points3d[1], points3d[2], "r*")
    plt.tight_layout()
    plt.show()


def compose_transformation_matrix(r11, r12, r13, r21, r22, r23, r31, r32, r33, tx, ty, tz):
    H = np.array([[r11, r12, r13, tx], [r21, r22, r23, ty], [r31, r32, r33, tz], [0, 0, 0, 1]])
    return H


def compose_projection_matrix(fx, fy, cx, cy, r11, r12, r13, r21, r22, r23, r31, r32, r33, tx, ty, tz):
    Mint = np.array([[fx, 0, cx, 0], [0, fy, cy, 0], [0, 0, 1, 0]])
    Mext = compose_transformation_matrix(r11, r12, r13, r21, r22, r23, r31, r32, r33, tx, ty, tz)
    P = Mint @ Mext
    return P, Mint, Mext


def compute_pixels_from_coln_eq_perfect(points3dInWorld, HCamToWorld):
    Xs = HCamToWorld[0, 3]
    Ys = HCamToWorld[1, 3]
    Zs = HCamToWorld[2, 3]

    r11 = HCamToWorld[0, 0]
    r12 = HCamToWorld[0, 1]
    r13 = HCamToWorld[0, 2]
    r21 = HCamToWorld[1, 0]
    r22 = HCamToWorld[1, 1]
    r23 = HCamToWorld[1, 2]
    r31 = HCamToWorld[2, 0]
    r32 = HCamToWorld[2, 1]
    r33 = HCamToWorld[2, 2]

    N = points3dInWorld.shape[1]
    uv = np.empty((2, N))
    for i in range(N):
        X = points3dInWorld[0, i]
        Y = points3dInWorld[1, i]
        Z = points3dInWorld[2, i]
        # careful with coordinate convention. it cause the sign inversion problem and wrong result
        # u = cx - fx * (r11 * (X - Xs) + r12 * (Y - Ys) + r13 * (Z - Zs)) / (r31 * (X - Xs) + r32 * (Y - Ys) + r33 * (Z - Zs))
        # v = cy - fy * (r21 * (X - Xs) + r22 * (Y - Ys) + r23 * (Z - Zs)) / (r31 * (X - Xs) + r32 * (Y - Ys) + r33 * (Z - Zs))
        u = cx + fx * (r11 * (X - Xs) + r12 * (Y - Ys) + r13 * (Z - Zs)) / (r31 * (X - Xs) + r32 * (Y - Ys) + r33 * (Z - Zs))
        v = cy + fy * (r21 * (X - Xs) + r22 * (Y - Ys) + r23 * (Z - Zs)) / (r31 * (X - Xs) + r32 * (Y - Ys) + r33 * (Z - Zs))
        uv[0, i] = u
        uv[1, i] = v
    return uv  # idealy perfect model, no dist, no noise


def generate_3d_position_inworld(N=30):
    points3d = np.empty((4, N), np.float32)
    points3d[0] = np.random.uniform(low=0.0, high=0.5, size=N)
    points3d[1] = np.random.uniform(low=0.0, high=0.5, size=N)
    points3d[2] = 0.0  # z flat
    points3d[3, :] = 1
    return points3d


def project_3d_position_to_camera_pixels(P, points3d):
    uvw = P @ points3d
    uv = uvw[:2, :] / uvw[2, :]
    # uvw[:2, :] += np.random.randn(2, N) * 1e-2  # add some noise
    return uv


def compute_error_objective(uvideal, uvmeasure):
    diffindv = uvideal - uvmeasure
    diffper = np.sum(diffindv**2, axis=0)
    error = np.sum(diffper)
    return error


def compute_mean_reproject_error(uvobserved, uvproject):
    diffindv = uvobserved - uvproject
    normper = np.linalg.norm(diffindv, axis=0)
    meanerror = np.sum(normper) / normper.shape[0]
    return meanerror


def compute_radial_distortion_radius(uv, cx, cy):
    center = np.array([[cx], [cy]])
    r = np.linalg.norm(uv - center, axis=0)
    return r


def apply_distortion(uvidl, cx, cy, k1, k2, k3, p1, p2):
    rs = compute_radial_distortion_radius(uvidl, cx, cy)

    uvdist = np.empty((2, rs.shape[0]))
    for i in range(rs.shape[0]):
        xidl = uvidl[0, i]
        yidl = uvidl[1, i]
        r = rs[i]
        udist = xidl * (1 + k1 * r**2 + k2 * r**4 + k3 * r**6)  # + (2 * p1 * xidl * yidl + p2 * (r**2 + 2 * xidl**2))
        print(f"> udist: {udist}")
        vdist = yidl * (1 + k1 * r**2 + k2 * r**4 + k3 * r**6)  # + (p1 * (r**2 + 2 * yidl**2) + 2 * p2 * xidl * yidl)
        print(f"> vdist: {vdist}")
        uvdist[0, i] = udist
        uvdist[1, i] = vdist
    return uvdist


if __name__ == "__main__":
    image_width = 640
    image_height = 480

    fx = 598.460339
    fy = 597.424060
    cx = 317.880979
    cy = 233.262422

    # distorstion k1, k2, p1, p2, k3 params order
    k1 = 0.000000000000002  # 0.142729
    k2 = 0.000000000000002  # -0.282139

    p1 = 0.000000000000002  # -0.005699
    p2 = 0.000000000000002  # -0.012027

    k3 = 0.0  # 0.000000

    # HWorldToCam
    r11 = 1
    r12 = 0
    r13 = 0
    r21 = 0
    r22 = 1
    r23 = 0
    r31 = 0
    r32 = 0
    r33 = 1
    tx = 0.0
    ty = 2
    tz = 2

    P, Mint, Mext = compose_projection_matrix(fx, fy, cx, cy, r11, r12, r13, r21, r22, r23, r31, r32, r33, tx, ty, tz)
    points3d = generate_3d_position_inworld(N=5)
    print(f"> points3d: {points3d}")
    uv = project_3d_position_to_camera_pixels(P, points3d)
    print(f"> uv: {uv}")

    HWorldToCam = compose_transformation_matrix(r11, r12, r13, r21, r22, r23, r31, r32, r33, tx, ty, tz)
    HCamToWorld = np.linalg.inv(HWorldToCam)
    uvn = compute_pixels_from_coln_eq_perfect(points3d, HCamToWorld)

    plot_stuff(Mext, points3d)

    uvdist = apply_distortion(uv, cx, cy, k1, k2, k3, p1, p2)
