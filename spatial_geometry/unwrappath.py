import os
import sys

sys.path.append(str(os.path.abspath(os.getcwd())))

import numpy as np
import matplotlib.pyplot as plt
from spatial_geometry.utils import Utils

# qinit = np.array([-2.0, -1.0]).T
path = np.array([[-2.0, -1.0], [-3.0, -1.0], [3.0, -1.0], [2.0, -1.0]]).T


plt.plot(path[0], path[1], "bo")
plt.show()

# qfrom = -2.0
# qto = -3.0
# t = 0.5

qfrom = -3.0
qto = 3.0
t = 1.0


def interpolate_so2(qfrom, qto, t):
    diff = qto - qfrom

    if abs(diff) <= np.pi:
        qnew = qfrom + diff * t

    else:
        if diff > 0.0:
            diff = 2.0 * np.pi - diff
        else:
            diff = -2.0 * np.pi - diff
        qnew = qfrom - diff * t

        # input states are within bounds, so the following check is sufficient
        # if we want to unwrap the value. maybe be we dont want to wrap it back
        if qnew > np.pi:
            qnew -= 2 * np.pi
        elif qnew < -np.pi:
            qnew += 2 * np.pi

    return qnew


def unwrappath(qfrom, qto):  # if we want to unwrap the value. maybe be we dont want to wrap it back
    t = 1.0  # it have to be 1.0
    diff = qto - qfrom

    if abs(diff) <= np.pi:
        qnew = qfrom + diff * t

    else:
        if diff > 0.0:
            diff = 2.0 * np.pi - diff
        else:
            diff = -2.0 * np.pi - diff
        qnew = qfrom - diff * t

    return qnew


# qnew = interpolate_so2(qfrom, qto, t)
# print(f"> qnew: {qnew}")


# qnewuw = unwrappath(qfrom, qto)
# print(f"> qnewuw: {qnewuw}")


path1d = np.array([-2.0, -3.0, 3.0, 2.0])
pathunwrap = [path1d[0]]
for i in range(path1d.shape[0] - 1):
    qnewuw = unwrappath(pathunwrap[i], path1d[i + 1])
    pathunwrap.append(qnewuw)
print(f"> pathunwrap: {pathunwrap}")


def norm(qa, qb):
    L = np.full_like(qa, 2 * np.pi)
    delta = np.abs(qa - qb)
    deltaw = L - delta
    deltat = np.min(np.hstack((delta, deltaw)), axis=1)
    return deltat, np.linalg.norm(deltat)

def interpolate_so2_multidim(qfrom, qto, t):
    diff = qto - qfrom
    print(f"> diff: {diff}")

    da = np.abs(diff)
    print(f"> da: {da}")

    dq = da <= np.pi
    print(f"> dq: {dq}")

    diff1 = 2.0 * np.pi - diff
    diff2 = -2.0 * np.pi - diff

    # if abs(diff) <= np.pi:
    #     qnew = qfrom + diff * t

    # else:
    #     if diff > 0.0:
    #         diff = 2.0 * np.pi - diff
    #     else:
    #         diff = -2.0 * np.pi - diff
    #     qnew = qfrom - diff * t

    #     # input states are within bounds, so the following check is sufficient
    #     # if we want to unwrap the value. maybe be we dont want to wrap it back
    #     if qnew > np.pi:
    #         qnew -= 2 * np.pi
    #     elif qnew < -np.pi:
    #         qnew += 2 * np.pi

    # return qnew


qfrom = np.array([[-3.0], [0.0]])
qto = np.array([[3.0], [0.0]])
t = 0.5
qnew = interpolate_so2_multidim(qfrom, qto, t)
print(f"> qnew: {qnew}")

deltat, n = norm(qfrom, qto)
print(f"> deltat: {deltat}")
print(f"> n: {n}")