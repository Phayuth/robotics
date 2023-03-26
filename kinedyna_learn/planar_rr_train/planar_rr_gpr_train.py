# https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.GaussianProcessRegressor.html
import os
import sys
wd = os.path.abspath(os.getcwd())
sys.path.append(str(wd))

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from robot.planar_rr import planar_rr

robot = planar_rr()

# # define dataset
# X, y = planar_rr_generate_dataset(robot)

# Define forward kinematics function
def forward_kinematics(q1, q2):
    x = 2*np.cos(q1) + 2*np.cos(q1 + q2)
    y = 2*np.sin(q1) + 2*np.sin(q1 + q2)
    return np.array([x, y])

# Define training data
X_train = np.array([[0, 0], [np.pi/4, np.pi/4], [np.pi/2, np.pi/2], [3*np.pi/4, np.pi], [np.pi, np.pi/2]])
y_train = np.array([forward_kinematics(q1, q2) for q1, q2 in X_train])

# Define kernel function
kernel = RBF(length_scale=1.0)

# Define Gaussian process model
model = GaussianProcessRegressor(kernel=kernel)

# Train model
model.fit(X_train, y_train)

# Define test data
X_test = np.array([[np.pi/3, np.pi/3], [np.pi/2, np.pi/3], [np.pi/3, np.pi/2]])

# Make predictions
y_pred, sigma = model.predict(X_test, return_std=True)

# Print results
print("Predicted end effector positions:\n", y_pred)
print("Uncertainties:\n", sigma)

theta_test = X_test[1].reshape(2,1)
robot.plot_arm(theta_test, plt_basis=True, plt_show=True)