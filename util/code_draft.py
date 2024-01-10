import os
import sys

wd = os.path.abspath(os.getcwd())
sys.path.append(str(wd))

import numpy as np
from icecream import ic


def ik_selectivly_dls():
    e = np.array([1, 3, 5]).reshape(3, 1)
    ic(e)

    Jac = np.array([[3, 4, 5], [1, 3, 2], [3, 6, 1]])
    ic(Jac)

    U, D, VT = np.linalg.svd(Jac)
    V = VT.T
    ic(U)
    ic(D)
    ic(VT)
    ic(V)

    Alpha = U.T @ e
    ic(Alpha)

    N = np.sum(Jac, axis=0)
    ic(N)

    M = np.array([4, 6, 1])
    ic(M)

    vabs = np.abs(V)
    ic(vabs)

    # NdivM = N/M
    # ic(NdivM)

    # gamma = np.minimum(3, NdivM)
    # ic(gamma)

ik_selectivly_dls()