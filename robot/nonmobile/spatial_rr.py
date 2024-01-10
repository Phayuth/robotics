import numpy as np
from spatial_geometry.spatial_transformation import RigidBodyTransformation as rbt


class SpatialRR:

    def __init__(self):
        self.alpha1 = np.pi / 2
        self.alpha2 = 0
        self.d1 = 1
        self.d2 = 0
        self.a1 = 0
        self.a2 = 1

    def forward_kinematic(self, theta):
        theta1 = theta[0, 0]
        theta2 = theta[1, 0]

        A1 = rbt.dh_transformation(theta1, self.alpha1, self.d1, self.a1)
        A2 = rbt.dh_transformation(theta2, self.alpha2, self.d2, self.a2)
        T = A1 @ A2

        p2 = np.array([[0], [0], [0], [1]])  # the fourth element MUST be equal to 1
        p0 = T @ p2

        return p0
