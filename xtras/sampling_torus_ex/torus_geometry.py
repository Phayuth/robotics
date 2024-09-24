import os
import sys

sys.path.append(str(os.path.abspath(os.getcwd())))

import matplotlib.pyplot as plt
import numpy as np

from spatial_geometry.utils import Utils

# qa = np.array([2.8, 1.0]).reshape(2, 1)
# qb = np.array([-2.8, 1.0]).reshape(2, 1)

qa = np.array([1.0, 1.0]).reshape(2, 1)
qb = np.array([-1.0, 1.0]).reshape(2, 1)

aa = Utils.minimum_dist_torus(qa, qb)
print(f"> aa: {aa}")


def steer(xFrom, xTo, distance):
    distI = xTo - xFrom
    dist = np.linalg.norm(distI)
    dI = (distI / dist) * distance
    newI = xFrom + dI
    return newI


def find_nearest_qb_to_qa(qa, qb, ignore_orginal=True):
    # if ignore_original there alway be torus path arround even the two point is close
    # if not, then the original will be consider and if it has minimum distance there only 1 way to move.
    limt2 = np.array([[-2 * np.pi, 2 * np.pi], [-2 * np.pi, 2 * np.pi]])
    Qb = Utils.find_alt_config(qb, limt2, filterOriginalq=ignore_orginal)
    di = Qb - qa
    n = np.linalg.norm(di, axis=0)
    minid = np.argmin(n)
    return Qb[:, minid, np.newaxis]


def steertorus(xFrom, xTo, distance):
    # incase of steer in torus, we need the new node to be nearest so we have to include the original qb
    candi = find_nearest_qb_to_qa(xFrom, xTo, ignore_orginal=False)
    newI = steer(xFrom, candi, distance)
    newI = Utils.wrap_to_pi(newI)
    return newI


eta = 0.4
qnew = steer(qa, qb, eta)
qnewt = steertorus(qa, qb, eta)

plt.plot([qa[0]], [qa[1]], "*b", label="qa")
plt.plot([qb[0]], [qb[1]], "or", label="qb")
plt.plot([qnewt[0]], qnewt[1], "*g", label="steer qa to qb torusly")
plt.plot([qnew[0]], qnew[1], ".k", label="steer qa to qb normally")
plt.xlim((-np.pi, np.pi))
plt.ylim((-np.pi, np.pi))
plt.legend()
plt.show()
