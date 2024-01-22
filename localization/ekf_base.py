import numpy as np


class EKFLocalization:

    def __init__(self, robotEKFModel) -> None:
        self.robotEKFModel = robotEKFModel
        self.Q = self.robotEKFModel.Q
        self.R = self.robotEKFModel.R
        self.PEst = np.eye(len(self.Q))

    def ekf_estimation(self, xEst, z, u, Ts):
        #  Predict
        xPred = self.robotEKFModel.motion_model(xEst, u, Ts)
        jF = self.robotEKFModel.jacob_f(xEst, u, Ts)
        PPred = jF @ self.PEst @ jF.T + self.Q

        #  Update
        zPred = self.robotEKFModel.observation_model(xPred)
        y = z - zPred
        jH = self.robotEKFModel.jacob_h()
        S = jH @ PPred @ jH.T + self.R
        K = PPred @ jH.T @ np.linalg.inv(S)
        xEst = xPred + K @ y
        self.PEst = (np.eye(len(xEst)) - K @ jH) @ PPred
        return xEst