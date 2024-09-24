import os
import sys

sys.path.append(str(os.path.abspath(os.getcwd())))

from spatial_geometry.spatial_transformation import RigidBodyTransformation as rbt

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


# ik_selectivly_dls()


# import json

# matrixsamples = np.random.uniform(0, 1, (4, 40)).tolist()
# jointsamples = np.random.uniform(-np.pi, np.pi, (10, 6)).tolist()
# matrix = np.random.uniform(0, 1, (4, 4)).tolist()
# pose = np.random.uniform(0, 1, (1, 7)).tolist()

# data = {"Title":"Handeye Calibration Program",
#         "Camera Mount Type": "Fixed",
#         "Camera Frame": "Camera_color_optical_frame",
#         "Object Frame": "calib_board",
#         "Robot Base": "base",
#         "End Effector Frame": "too0",
#         "Matrix Dataset": matrixsamples,
#         "Joint Dataset": jointsamples,
#         "Result Transformation": "Camera_color_optical_frame to base",
#         "Result Matrix": matrix,
#         "Result Pose": pose}


# with open("/home/yuth/jointsample.json", "w") as f:
#     json.dump(data, f, indent=4)

# with open("/home/yuth/jointsample.json", "r") as f:
#     a = json.load(f)

# print(a)


def stomp():
    import numpy as np
    import scipy

    def generate_A_matrix(order, N):
        A = np.zeros((N - order, N))
        for i in range(N - order):
            for j in range(order + 1):
                A[i, i + j] = (-1) ** j * scipy.special.comb(order, j)
        return A.T

    N = 10  # Number of waypoints
    order = 2  # Second-order finite difference
    A_matrix = generate_A_matrix(order, N)
    R = A_matrix.T @ A_matrix
    print(f"> R: {R}")


    # Example usage
    start = np.array([0, 0])
    goal = np.array([1, 1])
    initial_trajectory = np.linspace(start, goal, N)
    print(f"> initial_trajectory: {initial_trajectory}")


stomp()
