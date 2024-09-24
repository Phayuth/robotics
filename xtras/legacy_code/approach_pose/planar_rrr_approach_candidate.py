import os
import sys

sys.path.append(str(os.path.abspath(os.getcwd())))

import numpy as np
import matplotlib.pyplot as plt
from robot.nonmobile.planar_rrr import PlanarRRR
from spatial_geometry.spatial_shape import CoordinateTransform

# create robot instance
r = PlanarRRR()

# SECTION - create candidate pose
x_targ = 1.5
y_targ = 1.5
t_targ = np.linspace(np.pi / 2, 3 * np.pi / 2, 90)
r_inner = 0.1
r_outer = r_inner + 0.1
x_inner, y_inner = CoordinateTransform.polar_to_cartesian(r_inner, t_targ, x_targ, y_targ)
x_outer, y_outer = CoordinateTransform.polar_to_cartesian(r_outer, t_targ, x_targ, y_targ)

# SECTION - search for pose candidate
candidate = []
for i in range(len(x_inner)):

    x_ot = x_outer[i]
    y_ot = y_outer[i]

    a = [x_ot - x_targ, y_ot - y_targ]
    b = [1, 0]
    dot = np.dot(a, b)
    alpha_candidate = np.arccos(dot / (np.linalg.norm(a) * np.linalg.norm(b)))
    desired_angle_joint = alpha_candidate - np.pi

    # from each approach candidate, create desired pose
    desired_pose = np.array([[x_targ], [y_targ], [desired_angle_joint]])  # input x, y, phi
    try:
        # try to solve for ik for joint config
        inv_solu = r.inverse_kinematic_geometry(desired_pose)  # try to solve for inverse kinematic
    except:
        pass

    candidate.append(inv_solu)

# SECTION - plot task space
plt.axes().set_aspect('equal')
plt.axvline(x=0, c="black")
plt.axhline(y=0, c="black")
for i in range(len(x_inner)):
    plt.plot([x_inner[i], x_outer[i]], [y_inner[i], y_outer[i]], c="orange")
for i in candidate:
    r.plot_arm(i)
plt.show()



# SECTION - user define pose and calculate theta
x_targ = 0.5
y_targ = 0.5
alpha_targ = 2  # given from grapse pose candidate
d_app = 0.1
phi_targ = alpha_targ - np.pi
target = np.array([x_targ, y_targ, phi_targ]).reshape(3, 1)
t_targ = r.inverse_kinematic_geometry(target, elbow_option=0)


# SECTION - calculate approach pose and calculate theta
aprch = np.array([[d_app * np.cos(target[2, 0] + np.pi) + target[0, 0]], [d_app * np.sin(target[2, 0] + np.pi) + target[1, 0]], [target[2, 0]]])
t_app = r.inverse_kinematic_geometry(aprch, elbow_option=0)


# SECTION - plot task space
r.plot_arm(t_targ, plt_basis=True)
r.plot_arm(t_app)
plt.show()