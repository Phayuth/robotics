import numpy as np
import matplotlib.pyplot as plt
from icecream import ic

np.set_printoptions(linewidth=1000, suppress=True)


def hrz(theta):
    return np.array([[np.cos(theta), -np.sin(theta), 0, 0], [np.sin(theta), np.cos(theta), 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])


def ht(x, y, z):
    return np.array([[1, 0, 0, x], [0, 1, 0, y], [0, 0, 1, z], [0, 0, 0, 1]])


# obstacle
cobs = np.array([0.0, 1.5]).reshape(2, 1)  # x y
robs = 0.2  # r


def forward_points(thetas):
    l1 = 1
    l2 = 1
    Hj1ToB = hrz(thetas[0])
    Hj2Toj1 = ht(l1, 0, 0) @ hrz(thetas[1])

    P0123ToJ1 = np.array([[0, l1 / 3, 2 * l1 / 3, l1], [0, 0, 0, 0], [0, 0, 0, 0], [1, 1, 1, 1]])
    P456ToJ2 = np.array([[l2 / 3, 2 * l2 / 3, l2], [0, 0, 0], [0, 0, 0], [1, 1, 1]])

    P123ToB = Hj1ToB @ P0123ToJ1
    P456ToB = Hj1ToB @ Hj2Toj1 @ P456ToJ2

    pToB = np.hstack((P123ToB[0:2, ...], P456ToB[0:2, ...]))

    # plt.plot(pToB[0, ...], pToB[1, ...], "b*")
    # plt.xlim(-3, 3)
    # plt.ylim(-3, 3)
    # plt.show()

    return pToB


def compute_sdf(points):
    ec = points.T - cobs
    ccd = np.linalg.norm(ec, axis=1)
    rsum = np.full((points.shape[1],), 1 / 6) + robs
    sdf = ccd - rsum

    # plt.plot(points[0, ...], points[1, ...], "b*")
    # circle1 = plt.Circle(cobs, robs, color="r")
    # plt.gca().add_patch(circle1)
    # plt.xlim(-3, 3)
    # plt.ylim(-3, 3)
    # plt.show()
    return sdf


# thetas = np.array([0, 0])
# points = forward_points(thetas)
# ic(points)
# sdf = compute_sdf(points)
# ic(sdf)
# rb = 1/6
# ep = 0.1
# term = ep + rb - sdf
# ic(term)
# termext = np.vstack((term, np.zeros_like(term)))
# ic(termext)
# mm = np.max(termext, axis=0)
# ic(mm)
# q = mm.sum()
# ic(q)


def gen_trajectory():
    n = 10  # nwaypoints
    d = 2  # 2 joints
    # start = np.array([0] * d)
    # end = np.array([np.pi] * d)
    start = np.array([0, 0])
    end = np.array([np.pi, 0])

    xi = np.linspace(start, end, n, endpoint=True)
    ic(xi.shape)
    ic(xi)
    return xi


def hrz_array(theta):
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)

    hzz = np.array(
        [
            [cos_t, -sin_t, np.zeros_like(theta), np.zeros_like(theta)],
            [sin_t, cos_t, np.zeros_like(theta), np.zeros_like(theta)],
            [np.zeros_like(theta), np.zeros_like(theta), np.ones_like(theta), np.zeros_like(theta)],
            [np.zeros_like(theta), np.zeros_like(theta), np.zeros_like(theta), np.ones_like(theta)],
        ]
    )

    hzz = np.transpose(hzz, (2, 0, 1))  # Transpose the result to get shape (n, 4, 4)
    return hzz


def ht_array(x, y, z):
    htt = np.array(
        [
            [np.ones_like(x), np.zeros_like(x), np.zeros_like(x), x],
            [np.zeros_like(x), np.ones_like(x), np.zeros_like(x), y],
            [np.zeros_like(x), np.zeros_like(x), np.ones_like(x), z],
            [np.zeros_like(x), np.zeros_like(x), np.zeros_like(x), np.ones_like(x)],
        ]
    )

    htt = np.transpose(htt, (2, 0, 1))
    return htt


def forward_points_array(thetas):
    l1 = 1
    l2 = 1
    Hj1ToB = hrz_array(thetas[:, 0])
    Hj2Toj1 = ht_array(np.full(thetas.shape[0], l1), np.zeros(thetas.shape[0]), np.zeros(thetas.shape[0])) @ hrz_array(thetas[:, 1])

    P0123ToJ1 = np.array([[0, l1 / 3, 2 * l1 / 3, l1], [0, 0, 0, 0], [0, 0, 0, 0], [1, 1, 1, 1]])
    P456ToJ2 = np.array([[l2 / 3, 2 * l2 / 3, l2], [0, 0, 0], [0, 0, 0], [1, 1, 1]])

    P123ToB = Hj1ToB @ P0123ToJ1
    P456ToB = Hj1ToB @ Hj2Toj1 @ P456ToJ2

    pToB = np.concatenate((P123ToB, P456ToB), axis=2)

    # ic(Hj1ToB.shape)
    # ic(Hj2Toj1.shape)
    # ic(P0123ToJ1.shape)
    # ic(P456ToJ2.shape)
    # ic(P123ToB.shape)
    # ic(P456ToB.shape)
    # ic(pToB)
    # ic(pToB.shape)

    return pToB


def compute_traj_obscost_per_N(xi):
    pToB = forward_points_array(xi)
    pToBslice = pToB[:, 0:2, :]

    # minimum distance from bodies to obstacle
    ec = pToBslice - cobs
    ccd = np.linalg.norm(ec, axis=1)
    rsum = np.ones_like(ccd) * (1 / 6) + robs
    sdf = ccd - rsum

    # cost at 1 timestep of all bodies
    # rb = 1 / 6
    rb = 3 / 6
    ep = 0.1  # minimum padd from distance
    term = ep + rb - sdf
    termext = np.stack((term, np.zeros_like(term)), axis=-1)
    mm = np.max(termext, axis=2)  # cost per body
    q = mm.sum(axis=1)

    # return q
    return sdf.min(axis=1)


# xi = gen_trajectory()
# q = compute_traj_obscost_per_N(xi)
# ic(q)

# pToB = forward_points_array(xi)
# pToBslice = pToB[:, 0:2, :]

# ic(pToBslice)
# ic(pToBslice.shape)

# cobs = np.array([2, 0]).reshape(2, 1)  # x y
# robs = 0.5  # r

# ec = pToBslice - cobs
# ic(ec)

# ccd = np.linalg.norm(ec, axis=1)
# ic(ccd)
# ic(ccd.shape)

# rsum = np.ones_like(ccd) * (1 / 6) + robs
# ic(rsum)

# sdf = ccd - rsum
# ic(sdf)


# rb = 1 / 6
# ep = 0.1
# term = ep + rb - sdf
# ic(term)


# termext = np.stack((term, np.zeros_like(term)), axis=-1)
# ic(termext)

# mm = np.max(termext, axis=2)
# ic(mm)

# q = mm.sum(axis=1)
# ic(q)


def plot_arm(trajectory):
    circle1 = plt.Circle(cobs, robs, color="r")
    plt.gca().add_patch(circle1)

    for t in trajectory:
        pToB = forward_points(t)
        plt.plot(pToB[0, ...], pToB[1, ...], "b")

    plt.xlim(-3, 3)
    plt.ylim(-3, 3)
    plt.show()


# xi = gen_trajectory()
# plot_arm(xi)
