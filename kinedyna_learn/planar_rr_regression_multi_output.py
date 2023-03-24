import os
import sys
wd = os.path.abspath(os.getcwd())
sys.path.append(str(wd))

from sklearn.multioutput import MultiOutputRegressor
from sklearn.svm import LinearSVR
import numpy as np
from robot.planar_rr import planar_rr
from planar_rr_kinematic_dataset import planar_rr_generate_dataset

robot = planar_rr()

# define dataset
X, y = planar_rr_generate_dataset(robot)

# define base model
model = LinearSVR() # I think this model is not good for our application of robot arm that we already know the forward model involve sin and cos function

# define the direct multioutput wrapper model
wrapper = MultiOutputRegressor(model)

# fit the model on the whole dataset
wrapper.fit(X, y)

# make a single prediction
row = [0.5, 0.5] # end effector pose

# predict
yhat = wrapper.predict([row])
print("==>> yhat: \n", yhat)

theta_result = np.array([yhat]).reshape(2,1)
robot.plot_arm(theta_result, plt_basis=True, plt_show=True)