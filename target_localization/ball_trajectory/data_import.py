import os
import sys

wd = os.path.abspath(os.getcwd())
sys.path.append(str(wd))

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

from rigid_body_transformation.rotation_matrix import rotx, roty, rotz
from util.read_txt import read_txt_to_numpy

# Original data
parray = read_txt_to_numpy(txtFileString='target_localization/ball_trajectory/newposition.txt')
parray = parray[30:650,:]  # manual Trim

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(parray[:, 0], parray[:, 1], parray[:, 2], 'ro')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()

# Data manipulation, rotation, shift
ppp = parray.T
hx = rotx(-np.pi / 2)
hz = rotz(np.pi / 2)

pnew = hz @ hx @ ppp
pnew = pnew + np.array([1300, 400, 300]).reshape(3, 1)
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


# Define the parabolic equation
def parabolic(x, a, b, c):
    return a * x**2 + b * x + c


r = np.sqrt(x**2 + y**2)

# Fit the parabolic equation to the data
popt, pcov = curve_fit(parabolic, r, z)

# Print the values of a, b, and c
print("a = ", popt[0])
print("b = ", popt[1])
print("c = ", popt[2])

rreq = 900
zpred = parabolic(rreq, *popt)

# Plot the original data and the fitted curve
plt.plot(r, z, 'bo', label='Original Data')
plt.plot(r, parabolic(r, *popt), color='red', label='Fitted Curve')
plt.plot(r, parabolic(r, *popt), 'r*', label='Fitted Curve Descretize')

plt.plot(rreq, zpred, 'o', color="magenta")
plt.legend()
plt.xlabel('x')
plt.ylabel('y')
plt.show()