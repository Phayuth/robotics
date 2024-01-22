import numpy as np
from scipy.spatial.transform import Rotation as Rot


class DifferentialDriveEKFLocalizationModel:

    def __init__(self) -> None:
        self.Q = np.diag([0.1, 0.1, np.deg2rad(1.0)])**2 # Variance of location on x-axis, y-axis, yaw angle
        self.R = np.diag([1.0, 1.0])**2                  # Observation x,y position covariance

    def jacob_f(self, x, u, Ts):
        yaw = x[2, 0]
        v = u[0, 0]
        jF = np.array([[1.0, 0.0, -v * np.sin(yaw) * Ts],
                       [0.0, 1.0,  v * np.cos(yaw) * Ts],
                       [0.0, 0.0,                   1.0]])
        return jF

    def jacob_h(self):
        jH = np.array([[1, 0, 0],
                       [0, 1, 0]])
        return jH

    def motion_model(self, x, u, Ts):
        F = np.array([[1.0, 0.0, 0.0],
                      [0.0, 1.0, 0.0],
                      [0.0, 0.0, 1.0]])
        B = np.array([[Ts * np.cos(x[2, 0]),  0],
                      [Ts * np.sin(x[2, 0]),  0],
                      [                 0.0, Ts]])
        x = F @ x + B @ u
        return x

    def observation_model(self, x):
        H = np.array([[1.0, 0.0, 0.0],
                      [0.0, 1.0, 0.0]])
        z = H @ x
        return z

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


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from ekf_base import EKFLocalization

    robot = DifferentialDriveEKFLocalizationModel()
    ekf = EKFLocalization(robot)

    # State Vector [x y yaw]
    xEst = np.zeros((3, 1))
    xTrue = np.zeros((3, 1))
    xDR = np.zeros((3, 1))  # Dead reckoning

    # history
    histxEst = xEst
    histxTrue = xTrue
    histxDR = xTrue
    histz = np.zeros((2, 1))

    time = 0.0
    Ts = 0.1
    while 500 >= time:
        time += Ts
        if time < 100:
            v = 1.0
            omega = 0.2 * np.sin(0.05*time)
            u = np.array([[v], [omega]])
        elif 100 <= time <= 200:
            v = 1.0
            omega = 0.5 * np.sin(0.05*time) + np.cos(0.07*time)
            u = np.array([[v], [omega]])
        else:
            v = 1.0
            omega = 0.5 * np.sin(0.05*time) * np.cos(0.07*time)
            u = np.array([[v], [omega]])

        # find ground true state
        xTrue = robot.motion_model(xTrue, u, Ts)

        # simulate observation from sensor
        INPUT_NOISE = np.diag([1.0, np.deg2rad(30.0)])**2
        GPS_NOISE = np.diag([0.2, 0.2])**2
        z = robot.observation_model(xTrue) + GPS_NOISE @ np.random.randn(2, 1)  # add noise to gps x-y
        ud = u + INPUT_NOISE @ np.random.randn(2, 1)  # add noise to input
        xDR = robot.motion_model(xDR, ud, Ts) # observation for Dead reckoning

        # EKF Estimation
        xEst = ekf.ekf_estimation(xEst, z, ud, Ts)

        # store data history
        histxEst = np.hstack((histxEst, xEst))
        histxDR = np.hstack((histxDR, xDR))
        histxTrue = np.hstack((histxTrue, xTrue))
        histz = np.hstack((histz, z))

        plt.cla()
        plt.gcf().canvas.mpl_connect('key_release_event', lambda event: [exit(0) if event.key == 'escape' else None])
        plt.plot(histz[0, :], histz[1, :], ".g")
        plt.plot(histxTrue[0, :].flatten(), histxTrue[1, :].flatten(), "-b")
        plt.plot(histxDR[0, :].flatten(), histxDR[1, :].flatten(), "-k")
        plt.plot(histxEst[0, :].flatten(), histxEst[1, :].flatten(), "-r")
        DifferentialDriveEKFLocalizationModel.plot_covariance_ellipse(xEst, ekf.PEst)
        plt.axis("equal")
        plt.grid(True)
        plt.pause(0.001)