import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import HalfspaceIntersection

halfspaces = np.array([[-1.0, 0.0, 0.0], # -x + 0y <= 0
                       [0.0, -1.0, 0.0], # 0x -y <= 0
                       [2.0, 1.0, -4.0], # 2x +y <= -4
                       [-0.5, 1.0, -2.0]]) # -0.5x +y <= -2

feasible_point = np.array([0.5, 0.5]) # Interior Point: Must be strictly within the feasible region; otherwise, the algorithm won't work. init guess

hs = HalfspaceIntersection(halfspaces, feasible_point)
xintsect, yintsect = zip(*hs.intersections)

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1, aspect="equal")
xlim, ylim = (-1, 3), (-1, 3)
ax.set_xlim(xlim)
ax.set_ylim(ylim)

x = np.linspace(-1, 3, 100)
symbols = ["-", "+", "x", "*"]
signs = [0, 0, -1, -1]
fmt = {"color": None, "edgecolor": "b", "alpha": 0.5}

for h, sym, sign in zip(halfspaces, symbols, signs):
    hlist = h.tolist()
    fmt["hatch"] = sym
    if h[1] == 0:
        ax.axvline(-h[2] / h[0], label="{}x+{}y+{}=0".format(*hlist))
        xi = np.linspace(xlim[sign], -h[2] / h[0], 100)
        ax.fill_between(xi, ylim[0], ylim[1], **fmt)
    else:
        ax.plot(x, (-h[2] - h[0] * x) / h[1], label="{}x+{}y+{}=0".format(*hlist))
        ax.fill_between(x, (-h[2] - h[0] * x) / h[1], ylim[sign], **fmt)

ax.plot(xintsect, yintsect, "o", markersize=8)


# determine interior circle

from scipy.optimize import linprog
from matplotlib.patches import Circle

norm_vector = np.reshape(np.linalg.norm(halfspaces[:, :-1], axis=1), (halfspaces.shape[0], 1))
c = np.zeros((halfspaces.shape[1],))
c[-1] = -1
A = np.hstack((halfspaces[:, :-1], norm_vector))
b = -halfspaces[:, -1:]
res = linprog(c, A_ub=A, b_ub=b, bounds=(None, None))
x = res.x[:-1]
y = res.x[-1]
circle = Circle(x, radius=y, alpha=0.3)
ax.add_patch(circle)

plt.legend(bbox_to_anchor=(1.6, 1.0))
plt.show()
