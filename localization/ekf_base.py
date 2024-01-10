"""
EKF base class meant to be used with different type of mobile robot class

"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as Rot


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


def plot_covariance_ellipse(xEst, PEst):
    Pxy = PEst[0:2, 0:2]
    eigval, eigvec = np.linalg.eig(Pxy)

    if eigval[0] >= eigval[1]:
        bigind = 0
        smallind = 1
    else:
        bigind = 1
        smallind = 0

    t = np.arange(0, 2 * np.pi + 0.1, 0.1)
    a = np.sqrt(eigval[bigind])
    b = np.sqrt(eigval[smallind])
    x = a * np.cos(t)
    y = b * np.sin(t)
    angle = np.arctan2(eigvec[1, bigind], eigvec[0, bigind])
    rot = Rot.from_euler('z', angle).as_matrix()[0:2, 0:2]
    fx = rot @ (np.array([x, y]))
    px = np.array(fx[0, :] + xEst[0, 0]).flatten()
    py = np.array(fx[1, :] + xEst[1, 0]).flatten()
    plt.plot(px, py, "--r")
