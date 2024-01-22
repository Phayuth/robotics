import numpy as np


class SynchronusDrive:

    def __init__(self, wheelRadius, baseLength) -> None:
        # kinematic properties
        self.r = wheelRadius  #m
        self.d = baseLength   #m

    def forward_external_kinematic(self, bodyVelo, phi):
        # bodyVelo[0,0] = V and bodyVelo[1,0] = omega are controlled independently unlike diffdrive
        # so no internal kinematic, just read from sensor directly
        J = np.array([[np.cos(phi), 0],
                      [np.sin(phi), 0],
                      [          0, 1]])
        globalVelo = J @ bodyVelo
        return globalVelo