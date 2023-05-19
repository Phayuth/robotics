import os
import sys

wd = os.path.abspath(os.getcwd())
sys.path.append(str(wd))

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

from robot.ur5e import UR5e

# Define the parabolic equation
def zz(r):
    a =  -0.5644395478938202
    b =  4.9956333931451
    c =  -2.617093465831485
    return a*r**2 + b*r + c

x = np.linspace(5,0,100)
y = np.linspace(5,0,100)
r = np.sqrt(x**2 + y**2)
z = zz(r)




# pred
xreq = 0.5
yreq = 0.5
rreq = np.sqrt(xreq**2 + yreq**2)
zpred = zz(rreq)
print(f"==>> zpred: \n{zpred}")




# SECTION - create class robot
robot = UR5e()

TOriginal = np.array([[1, 0, 0, xreq],
                      [0, 1, 0, yreq],
                      [0, 0, 1, zpred],
                      [0, 0, 0, 1]])

# SECTION - inverse kinematic
thetaIk = robot.inverse_kinematic_geometry(TOriginal)
print("==>> thetaIk: \n", thetaIk.T)

thetaDecided = thetaIk[:,0]
print(f"==>> thetaDecided: \n{thetaDecided}")





# Plot the parabolic trajectory
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
# ax.plot3D([0, 2], [0, 0], [0, 0], 'red', linewidth=4)
# ax.plot3D([0, 0], [0, 2], [0, 0], 'purple', linewidth=4)
# ax.plot3D([0, 0], [0, 0], [0, 2], 'green', linewidth=4)
robot.plot_arm(thetaDecided.reshape(6,1), plt_basis=True, ax=ax)
ax.plot(x, y, z, label='Parabolic Trajectory')
ax.plot(xreq, yreq, zpred, 'ro')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.legend()

plt.show()