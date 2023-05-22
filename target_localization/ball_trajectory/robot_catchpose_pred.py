import os
import sys

wd = os.path.abspath(os.getcwd())
sys.path.append(str(wd))

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

from robot.ur5e import UR5e


# Define the parabolic equation
def pred(r):
    a = -0.5644395478938202
    b = 4.9956333931451
    c = -2.617093465831485
    return a * r**2 + b * r + c


x = np.linspace(5, 0, 100)
y = np.linspace(5, 0, 100)
r = np.sqrt(x**2 + y**2)
z = pred(r)

# pred
xReq = 0.5
yReq = 0.5
rReq = np.sqrt(xReq**2 + yReq**2)
zPred = pred(rReq)
print(f"==>> zPred: \n{zPred}")

# robot
robot = UR5e()
TOriginal = np.array([[1, 0, 0, xReq], [0, 1, 0, yReq], [0, 0, 1, zPred], [0, 0, 0, 1]])
thetaIk = robot.inverse_kinematic_geometry(TOriginal)
thetaDecided = thetaIk[:, 0]
print("==>> thetaIk: \n", thetaIk.T)
print(f"==>> thetaDecided: \n{thetaDecided}")

# Plot the parabolic trajectory
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
robot.plot_arm(thetaDecided.reshape(6, 1), plt_basis=True, ax=ax)
ax.plot(x, y, z, label='Parabolic Trajectory')
ax.plot(xReq, yReq, zPred, 'ro')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.legend()

plt.show()