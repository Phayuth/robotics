import os
import sys

sys.path.append(str(os.path.abspath(os.getcwd())))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Ellipse
from scipy.spatial.transform import Rotation as Rot
from spatial_geometry.utils import Utils
from matplotlib import collections as mc
from scipy.spatial import ConvexHull
from matplotlib.backend_bases import MouseButton
from shapely.geometry import Point as ShapelyPoint
from shapely.geometry import Polygon as ShapelyPolygon
from icecream import ic

np.set_printoptions(precision=3, suppress=True, linewidth=2000)


def get_reduced_space(qgoal, limt):
    Qgoalset = Utils.find_alt_config(qgoal.reshape(-1, 1), limt)
    mimm = np.min(Qgoalset, axis=1)
    maxx = np.max(Qgoalset, axis=1)
    limitspace = np.array([mimm, maxx]).T
    return limitspace  # [min max]


def get_new_init_in_reduced_space(qinit, limt, space):
    Qinitset = Utils.find_alt_config(qinit.reshape(-1, 1), limt)

    for i in range(Qinitset.shape[1]):
        q = Qinitset[:, i]
        q = q.reshape(-1, 1)
        if np.all(q >= space[:, 0]) and np.all(q <= space[:, 1]):
            return q

    return None


def correct_path(path, qinit_original):

    pass


if __name__ == "__main__":
    # limits
    limt2 = np.array([[-2 * np.pi, 2 * np.pi], [-2 * np.pi, 2 * np.pi]])

    qinit = np.array([0.0, 0.0])
    qgoal = np.array([4.0, 4.0])

    newspace = get_reduced_space(qgoal, limt2)
    ic(newspace)

    newinit = get_new_init_in_reduced_space(qinit, limt2, newspace)
    ic(newinit)
