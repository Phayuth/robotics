import os
import sys
wd = os.path.abspath(os.getcwd())
sys.path.append(str(wd))

from sklearn import svm
import numpy as np
from map.taskmap_geo_format import task_rectangle_obs_1
from robot.planar_rr import PlanarRR
from planar_rr_collision_dataset import collision_dataset
import matplotlib.pyplot as plt

robot = PlanarRR()
obs_list = task_rectangle_obs_1()

# Generate some training data
# X = np.array([[0, 0], [1, 1], [1, 0], [0, 1]])
# y = np.array([0, 0, 1, 1])
X, y = collision_dataset(robot, obs_list)

plt.imshow(y.reshape(180, 180))
plt.show()

# Train an SVM model
clf = svm.SVC(kernel='rbf', C=1)
clf.fit(X, y)

# Use the trained model to predict new data
print(clf.predict([[2, 2]]))

sample_size = 180
theta_candidate = np.linspace(-np.pi, np.pi, sample_size)

map = np.ones((180,180))

for ind1, th1 in enumerate(theta_candidate):
    for ind2, th2 in enumerate(theta_candidate):
        collision = clf.predict([[th1, th2]])
        map[ind1, ind2] = int(collision)

plt.imshow(map)
plt.show()
