import numpy as np


class AckermanSteer:

    def __init__(self, wheelRadius, baseLength) -> None:
        self.r = wheelRadius  #m
        self.L = baseLength  #m

    def forward_internal_kinematic(self, wheelVelo):  # wheelVelo is rotation omega
        #return np.array([linearVelo, angularVelo]).reshape(2, 1)
        raise NotImplementedError

    def forward_external_kinematic(self, motionInput, theta):  # motionInput is V and alpha
        return np.array([[motionInput[0, 0] * np.cos(theta)],
                         [motionInput[0, 0] * np.sin(theta)],
                         [(motionInput[0, 0] / self.L) * np.tan(motionInput[1, 0])]])

    def physical_limit(self, motionInput):  # physical limitations of the vehicle
        v = motionInput[0, 0]
        alpha = motionInput[1, 0]
        if abs(alpha) > np.pi / 4:
            alpha = np.pi / 4 * np.sign(alpha)
        if abs(v) > 0.8:
            v = 0.8 * np.sign(v)
        return np.array([[v], [alpha]])