import os
import sys

wd = os.path.abspath(os.getcwd())
sys.path.append(str(wd))

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

from rigid_body_transformation.rotation_matrix import rotx, roty, rotz
from util.read_txt import read_txt_to_numpy

# Original data
parray = read_txt_to_numpy(txtFileString='target_localization/ball_trajectory/newposition.txt')
parray = parray[30:650, :]  # manual Trim
ppp = parray.T
hx = rotx(-np.pi / 2)
hz = rotz(np.pi / 2)
pnew = hz @ hx @ ppp
pnew = pnew.T
x = pnew[:, 0]
y = pnew[:, 1]
z = pnew[:, 2]

# Create polynomial features up to degree 2
poly_features = PolynomialFeatures(degree=2, include_bias=True)
X = poly_features.fit_transform(np.column_stack((x, y)))

# Create and fit the regression model
model = LinearRegression()
model.fit(X, z)

# Get the regression coefficients
intercept = model.intercept_
coefficients = model.coef_

print("Intercept (w0):", intercept)
print("Coefficients (w1, w2, w3, w4, w5):", coefficients)

# Generate a grid of points for x and y
x_range = np.linspace(min(x), max(x), 100)
y_range = np.linspace(min(y), max(y), 100)
X_grid, Y_grid = np.meshgrid(x_range, y_range)

# Compute the predicted z values for each point on the grid
Z_grid = intercept + coefficients[1] * X_grid + coefficients[2] * Y_grid + coefficients[3] * X_grid**2 + coefficients[4] * Y_grid**2 + coefficients[5] * X_grid * Y_grid

# Create the 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x, y, z, c='b', label='Actual Data')
ax.plot3D([0, 200], [0, 0], [0, 0], 'red', linewidth=4)
ax.plot3D([0, 0], [0, 200], [0, 0], 'purple', linewidth=4)
ax.plot3D([0, 0], [0, 0], [0, 200], 'green', linewidth=4)
ax.plot_surface(X_grid, Y_grid, Z_grid, cmap='viridis', alpha=0.5, label='Regression Surface')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.set_title('Quadratic Regression')

# Show the plot
plt.show()
