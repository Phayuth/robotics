"""
Tricycle Robot
"""
import numpy as np


class TricycleDrive:

    def __init__(self, wheelRadius, baseLength) -> None:
        self.r = wheelRadius  #m
        self.L = baseLength  #m

        # physical limit
        self.linearVeloLimit = 0.8
        self.alphaLimit = np.pi / 4

    def forward_internal_kinematic(self, wheelVelo):  # wheelVelo is rotation omega
        raise NotImplementedError
        #return np.array([linearVelo, angularVelo]).reshape(2, 1)

    def forward_external_kinematic(self, motionInput, theta): # motionInput is V and alpha
        return np.array([[motionInput[0, 0] * np.cos(theta)],
                         [motionInput[0, 0] * np.sin(theta)],
                         [(motionInput[0, 0] / self.L) * np.tan(motionInput[1, 0])]])

    def physical_limit(self, motionInput):  # physical limitations of the vehicle
        v = motionInput[0, 0]
        alpha = motionInput[1, 0]

        if abs(v) > self.linearVeloLimit:
            v = self.linearVeloLimit * np.sign(v)
        if abs(alpha) > self.alphaLimit:
            alpha = self.alphaLimit * np.sign(alpha)

        return np.array([[v], [alpha]])