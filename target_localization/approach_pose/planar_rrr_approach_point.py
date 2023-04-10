import os
import sys

wd = os.path.abspath(os.getcwd())
sys.path.append(str(wd))

import matplotlib.pyplot as plt
import numpy as np
from util.coord_transform import polar2cats, circle_plt
from robot.planar_rrr import PlanarRRR

# create robot instance
robot = PlanarRRR()


# SECTION - user define pose and calculate theta
x_targ = 0.5
y_targ = 0.5
alpha_targ = 2  # given from grapse pose candidate
d_app = 0.1
phi_targ = alpha_targ - np.pi
target = np.array([x_targ, y_targ, phi_targ]).reshape(3, 1)
t_targ = robot.inverse_kinematic_geometry(target, elbow_option=0)


# SECTION - calculate approach pose and calculate theta
aprch = np.array([[d_app * np.cos(target[2, 0] + np.pi) + target[0, 0]], [d_app * np.sin(target[2, 0] + np.pi) + target[1, 0]], [target[2, 0]]])
t_app = robot.inverse_kinematic_geometry(aprch, elbow_option=0)


# SECTION - plot task space
robot.plot_arm(t_targ, plt_basis=True)
robot.plot_arm(t_app)
circle_plt(x_targ, y_targ, d_app)
plt.show()
