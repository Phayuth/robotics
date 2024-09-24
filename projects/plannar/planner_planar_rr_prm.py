import os
import sys

sys.path.append(str(os.path.abspath(os.getcwd())))

import numpy as np

np.random.seed(9)

import matplotlib.pyplot as plt
from planner.sampling_based.prm_base import PRMBase
from planner.sampling_based.prm_component import PRMPlotter
from simulator.sim_planar_rr import RobotArm2DSimulator
from spatial_geometry.utils import Utils

robotsim = RobotArm2DSimulator(torusspace=True)
prm = PRMBase(simulator=robotsim, eta=0.3, subEta=0.05)
prm.build_graph(4 * 1000)
prm.save_graph("prmgh")
prm.load_graph("prmgh")

# single search
xStart = np.array([0.0, 0.0]).reshape(2, 1)
xGoal = np.array([-2.3, 2.3]).reshape(2, 1)

path = prm.query(xStart, xGoal, prm.nodes, searcher="ast")
print(f"> path: {path}")

fig, ax = plt.subplots()
ax.set_axis_off()
fig.set_size_inches(w=3.40067, h=3.40067)
fig.tight_layout()
plt.xlim((-2 * np.pi, 2 * np.pi))
plt.ylim((-2 * np.pi, 2 * np.pi))
PRMPlotter.plot_2d_complete(path, prm, ax)
plt.show()


# multiple search
xGoal = np.array([5.90, -5.90]).reshape(2, 1)
limt2 = np.array([[-2 * np.pi, 2 * np.pi], [-2 * np.pi, 2 * np.pi]])
xg = Utils.find_alt_config(xGoal, limt2)
xGoals = [xg[:, i, np.newaxis] for i in range(xg.shape[1])]

path = prm.query_multiple_goal(xStart, xGoals, prm.nodes, searcher="dij")
print(f"> path: {path}")

fig, ax = plt.subplots()
ax.set_axis_off()
fig.set_size_inches(w=3.40067, h=3.40067)
fig.tight_layout()
plt.xlim((-2 * np.pi, 2 * np.pi))
plt.ylim((-2 * np.pi, 2 * np.pi))
PRMPlotter.plot_2d_complete(path, prm, ax)
plt.show()
