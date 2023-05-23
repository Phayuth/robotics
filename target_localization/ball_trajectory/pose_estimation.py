import os
import sys

wd = os.path.abspath(os.getcwd())
sys.path.append(str(wd))

import matplotlib.pyplot as plt
import numpy as np

from rigid_body_transformation.rotation_matrix import rotx, roty, rotz
from util.read_txt import read_txt_to_numpy
from parabolic_equation import parabolic

# Real Dataset
parray = read_txt_to_numpy(txtFileString='target_localization/ball_trajectory/newposition.txt')
parray = parray[30:650, :]  # manual Trim
ppp = parray.T
hx = rotx(-np.pi / 2)
hz = rotz(np.pi / 2)
pnew = hz @ hx @ ppp
pnew = pnew.T
xPose = pnew[:, 0]
yPose = pnew[:, 1]
zPose = pnew[:, 2]
t = np.linspace(0,1,xPose.shape[0])

# Generate Dataset
# fit param
xParam = [-2706, 6003, 329.6]
yParam = [368.4, 45.34, 258.9]
zParam = [-20530, 1866, 466.5]
testParam = [-0.5644395478938202, 4.9956333931451, -2.617093465831485]

# func of time
# t = np.linspace(0, 0.5, 100)
# xPose = parabolic(t,xParam[0],xParam[1],xParam[2])
# yPose = parabolic(t,yParam[0],yParam[1],yParam[2])
# zPose = parabolic(t,zParam[0],zParam[1],zParam[2])

# func of r
# t = np.linspace(0, 5, 100)
# xPose = np.linspace(-2, 0, 100)
# yPose = np.linspace(-1, 0, 100)
# rPose = np.sqrt(xPose**2 + yPose**2)
# zPose = parabolic(rPose, testParam[0], testParam[1], testParam[2])

# Pose Trajectory in Time
fig, axs = plt.subplots(3, 1)
axs[0].plot(t, xPose)
axs[1].plot(t, yPose)
axs[2].plot(t, zPose)
plt.show()

# 3D Trajectory
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(xPose, yPose, zPose)
ax.plot3D([0, 200], [0, 0], [0, 0], 'red', linewidth=4)
ax.plot3D([0, 0], [0, 200], [0, 0], 'purple', linewidth=4)
ax.plot3D([0, 0], [0, 0], [0, 200], 'green', linewidth=4)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.legend()

plt.show()