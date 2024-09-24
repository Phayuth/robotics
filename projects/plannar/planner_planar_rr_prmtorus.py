import os
import sys

sys.path.append(str(os.path.abspath(os.getcwd())))

import numpy as np

np.random.seed(9)

import matplotlib.pyplot as plt
from planner.sampling_based_torus.prm_base import PRMTorusRedundantBase
from planner.sampling_based_torus.prm_component import PRMTorusRedundantPlotter
from simulator.sim_planar_rr import RobotArm2DSimulator

robotsim = RobotArm2DSimulator(torusspace=False)
prm = PRMTorusRedundantBase(simulator=robotsim, eta=0.3, subEta=0.05)
# prm.build_graph(2000)
# prm.save_graph("datasave/prmroadmap/prm_g3")
prm.load_graph("datasave/prmroadmap/prm_g3")

# # single search
# xStart = np.array([0.0, 0.0]).reshape(2, 1)
# xGoal = np.array([-2.3, 2.3]).reshape(2, 1)

xStart = np.array([0.0, 2.8]).reshape(2, 1)
xGoal = np.array([-1.2, -2.3]).reshape(2, 1)

path = prm.query(xStart, xGoal, prm.nodes, searcher="dij") # have to use simple dijkstra for now cause astar have heuristic to make path not wrap
print(f"> path: {path}")


fig, ax = plt.subplots()
ax.set_axis_off()
fig.set_size_inches(w=1.5*3.40067, h=1.5*3.40067)
fig.tight_layout()
plt.xlim((-np.pi, np.pi))
plt.ylim((-np.pi, np.pi))
PRMTorusRedundantPlotter.plot_2d_complete(path, prm, ax)
plt.show()
