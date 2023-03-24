# https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.GaussianProcessRegressor.html

import os
import sys
wd = os.path.abspath(os.getcwd())
sys.path.append(str(wd))

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from planar_rr_kinematic_dataset import planar_rr_generate_dataset
from robot.planar_rr import planar_rr

robot = planar_rr()

# define dataset
X, y = planar_rr_generate_dataset(robot)

kernel = 1 * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))
gaussian_process = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9)
gaussian_process.fit(X, y)

# make a single prediction
row = [0.5, 0.5] # end effector pose

# predict
yhat = gaussian_process.predict([row])
# summarize the prediction
print('Predicted: %s' % yhat[0])

theta_result = np.array([yhat]).reshape(2,1)
robot.plot_arm(theta_result, plt_basis=True, plt_show=True)