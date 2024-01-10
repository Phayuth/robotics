"""
Omnidirectional Drive Robot
Reference : WMR book P30

"""
import numpy as np


class OmniDrive:

    def __init__(self, baseRadius) -> None:
        self.R = baseRadius
        self.theta2 = np.deg2rad(120)
        self.theta3 = np.deg2rad(240)

    def forward_external_kinematic(self, wheelVelo, phi):
        S = np.array([[-np.sin(phi), -np.sin(phi + self.theta2), -np.sin(phi + self.theta3)], 
                      [ np.cos(phi),  np.cos(phi + self.theta2),  np.cos(phi + self.theta3)], 
                      [  1/2*self.R,                 1/2*self.R,                 1/2*self.R]])
        globalVelo = ((2/3) * S) @ wheelVelo
        return globalVelo

    def inverse_external_kinematic(self, gobalVelo, phi):
        J = np.array([[              -np.sin(phi),               np.cos(phi), self.R], 
                      [-np.sin(phi + self.theta2), np.cos(phi + self.theta2), self.R], 
                      [-np.sin(phi + self.theta3), np.cos(phi + self.theta3), self.R]])
        wheelVelo = J @ gobalVelo
        return wheelVelo