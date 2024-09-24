import os
import sys

sys.path.append(str(os.path.abspath(os.getcwd())))

import numpy as np
np.random.seed(9)

import matplotlib.pyplot as plt
from planner.sampling_based.prm_base import PRMBase
from planner.sampling_based.prm_component import PRMPlotter
from simulator.sim_rectangle import TaskSpace2DSimulator

robotsim = TaskSpace2DSimulator()
prm = PRMBase(simulator=robotsim, eta=0.3, subEta=0.05)
prm.build_graph(1000)

xStart = np.array([0.0, 0.0]).reshape(2, 1)
xGoal = np.array([-2.3, 2.3]).reshape(2, 1)
path = prm.query(xStart, xGoal, prm.nodes, searcher="ast")
print(f"> path: {path}")

fig, ax = plt.subplots()
ax.set_axis_off()
fig.set_size_inches(w=3.40067, h=3.40067)
fig.tight_layout()
plt.xlim((-np.pi, np.pi))
plt.ylim((-np.pi, np.pi))
PRMPlotter.plot_2d_config(path, prm, ax)
plt.show()
