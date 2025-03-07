import os
import sys

sys.path.append(str(os.path.abspath(os.getcwd())))

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

from spatial_geometry.spatial_transformation import RigidBodyTransformation as rbt
from helper import read_txt_to_numpy


# Original data
parray = read_txt_to_numpy(txtFileString='datasave/joint_value/object_path.txt')
parray = parray[30:650, :]  # manual Trim

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(parray[:, 0], parray[:, 1], parray[:, 2], 'ro')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()

# Data manipulation, rotation, shift
ppp = parray.T
hx = rbt.rotx(-np.pi / 2)
hz = rbt.rotz(np.pi / 2)

pnew = hz @ hx @ ppp
pnew = pnew.T

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(pnew[:, 0], pnew[:, 1], pnew[:, 2], 'ro')
ax.plot3D([0, 200], [0, 0], [0, 0], 'red', linewidth=4)
ax.plot3D([0, 0], [0, 200], [0, 0], 'purple', linewidth=4)
ax.plot3D([0, 0], [0, 0], [0, 200], 'green', linewidth=4)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()

x = pnew[:, 0]
y = pnew[:, 1]
z = pnew[:, 2]
t = np.linspace(0,1,pnew.shape[0])

# Define the parabolic equation
def parabolic(x, a, b, c):
    return a * x**2 + b * x + c

# Fit the line equation
poptX, pcovX = curve_fit(parabolic, t, x)
poptY, pcovY = curve_fit(parabolic, t, y)
poptZ, pcovZ = curve_fit(parabolic, t, z)


fig1 = plt.figure("X")
ax1 = fig1.add_subplot(111)
ax1.plot(t,x, "ro")
ax1.plot(t, parabolic(t, *poptX))

fig2 = plt.figure("Y")
ax2 = fig2.add_subplot(111)
ax2.plot(t,y,"ro")
ax2.plot(t, parabolic(t, *poptY))

fig3 = plt.figure("Z")
ax3 = fig3.add_subplot(111)
ax3.plot(t,z,"ro")
ax3.plot(t, parabolic(t, *poptZ))
plt.show()


tReq = 1.3
xPred = parabolic(tReq, *poptX)
yPred = parabolic(tReq, *poptY)
zPred = parabolic(tReq, *poptZ)

fig1 = plt.figure("X")
ax1 = fig1.add_subplot(111)
ax1.plot(t,x, "ro")
ax1.plot(t, parabolic(t, *poptX))
ax1.plot(tReq, xPred, "g*")

fig2 = plt.figure("Y")
ax2 = fig2.add_subplot(111)
ax2.plot(t,y,"ro")
ax2.plot(t, parabolic(t, *poptY))
ax2.plot(tReq, yPred, "g*")

fig3 = plt.figure("Z")
ax3 = fig3.add_subplot(111)
ax3.plot(t,z,"ro")
ax3.plot(t, parabolic(t, *poptZ))
ax3.plot(tReq, zPred, "g*")

plt.show()