"""
Forward Kinematic Model Train using Gaussian Process Regression for Planar RR
https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.GaussianProcessRegressor.html
"""

import os
import sys

sys.path.append(str(os.path.abspath(os.getcwd())))

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF


def forward_kinematics(q1, q2):
    x = 2 * np.cos(q1) + 2 * np.cos(q1 + q2)
    y = 2 * np.sin(q1) + 2 * np.sin(q1 + q2)
    return np.array([x, y])


# Define training data
X_train = np.array([[0, 0], [np.pi / 4, np.pi / 4], [np.pi / 2, np.pi / 2], [3 * np.pi / 4, np.pi], [np.pi, np.pi / 2]])
y_train = np.array([forward_kinematics(q1, q2) for q1, q2 in X_train])

kernel = RBF(length_scale=1.0)
model = GaussianProcessRegressor(kernel=kernel)
model.fit(X_train, y_train)

# Define test data
X_test = np.array([[np.pi / 3, np.pi / 3], [np.pi / 2, np.pi / 3], [np.pi / 3, np.pi / 2]])
y_pred, sigma = model.predict(X_test, return_std=True)

# Print results
print("Predicted end effector positions:\n", y_pred)
print("Uncertainties:\n", sigma)

theta_test = X_test[1].reshape(2, 1)
print(f"> theta_test: {theta_test}")
