"""
Support Vector machine for classify 2 class of data
https://towardsdatascience.com/understanding-the-hyperplane-of-scikit-learns-svc-model-f8515a109222
"""

from sklearn import svm
from sklearn.inspection import DecisionBoundaryDisplay
import numpy as np
import matplotlib.pyplot as plt

# we create 20 points seperable in two labels
X = np.array(
    [
        [-1.0, 2.0],
        [-1.5, 2.0],
        [-2.0, 2.0],
        [-1.5, 1.0],
        [-1.5, 1.5],
        [-1.0, 1.5],
        [-1.0, 2.0],
        [2.0, -2.0],
        [2.5, -2.0],
        [3.0, -2.0],
        [2.0, -1.0],
        [2.0, -1.5],
        [2.5, -1.5],
        [3.0, -1.5],
    ]
)

y = [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]

# svm
clf = svm.SVC(kernel="linear")
clf.fit(X, y)

fig, ax = plt.subplots()
ax.scatter(X[:, 0], X[:, 1], c=y, s=20, label="training points")
ax.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=100, linewidth=1, facecolors="none", edgecolors="k", label="support vectors")
DecisionBoundaryDisplay.from_estimator(clf, X, plot_method="contour", colors="k", levels=[-1, 0, 1], alpha=0.5, linestyles=["--", "-", "--"], ax=ax)

new_point_1 = np.array([[-1.0, 2.5]])
new_point_2 = np.array([[2, -2.5]])
ax.scatter(new_point_1[:, 0], new_point_1[:, 1], c="blue", s=20, label="new_point_1")
ax.scatter(new_point_2[:, 0], new_point_2[:, 1], c="red", s=20, label="new_point_2")
ax.legend()
plt.show()

print(f"svm coefficients, intercepts: {clf.coef_}, {clf.intercept_}")


print(clf.predict(new_point_1))
print(clf.predict(new_point_2))

# manual calculation (the same as using predict method)
print(np.dot(clf.coef_[0], new_point_1[0]) + clf.intercept_)
print(np.dot(clf.coef_[0], new_point_2[0]) + clf.intercept_)
