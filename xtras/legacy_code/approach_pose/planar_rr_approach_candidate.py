import os
import sys

wd = os.path.abspath(os.getcwd())
sys.path.append(str(wd))

import numpy as np
import matplotlib.pyplot as plt
from robot.nonmobile.planar_rr import PlanarRR
from spatial_geometry.taskmap_geo_format import NonMobileTaskMap
from spatial_geometry.spatial_shape import CoordinateTransform

# create robot instance
robot = PlanarRR()

# SECTION - user define pose
x_targ = 1.5
y_targ = 1.5
t_targ = np.linspace(np.pi / 2, 3 * np.pi / 2, 10)  # candidate pose
app_d = 0.5
app_x, app_y = CoordinateTransform.polar_to_cartesian(app_d, t_targ, x_targ, y_targ)
obs_list = NonMobileTaskMap.task_rectangle_obs_1()

# SECTION - calculate ik for each approach pose and the main pose
t_app = [robot.inverse_kinematic_geometry(np.array([[app_x[i]], [app_y[i]]]), elbow_option=0) for i in range(len(app_x))]
t_app_main = robot.inverse_kinematic_geometry(np.array([[x_targ], [y_targ]]), elbow_option=0)

# SECTION - plot task space
plt.axes().set_aspect('equal')
plt.axvline(x=0, c="black")
plt.axhline(y=0, c="black")
plt.plot(app_x, app_y, c="green")
for obs in obs_list:
    obs.plot()
for each_theta in range(len(t_app)):
    robot.plot_arm(np.array([[t_app[each_theta][0].item()], [t_app[each_theta][1].item()]]))
plt.show()


# SECTION - define target pose and calculate theta
x_targ = 1.6
y_targ = 2.15
target = np.array([x_targ, y_targ]).reshape(2, 1)
t_targ = robot.inverse_kinematic_geometry(target, elbow_option=0)


# SECTION - calculate approach point and calculate theta
d_app = 0.3
alpha = np.pi + np.sum(t_targ)
aprch = np.array([d_app * np.cos(alpha), d_app * np.sin(alpha)]).reshape(2, 1) + target
t_app = robot.inverse_kinematic_geometry(aprch, elbow_option=0)


# SECTION - plot task space
obs_list = task_rectangle_obs_1()
robot.plot_arm(t_app, plt_basis=True)
robot.plot_arm(t_targ)
for obs in obs_list:
    obs.plot()
plt.show()