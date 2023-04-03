import os
import sys
wd = os.path.abspath(os.getcwd())
sys.path.append(str(wd))

import numpy as np
import matplotlib.pyplot as plt
from robot.planar_rrr import planar_rrr
from util.coord_transform import polar2cats, circle_plt

# create robot instance
r = planar_rrr()

# SECTION - create candidate pose
x_targ = 1.5
y_targ = 1.5
t_targ = np.linspace(np.pi/2,3*np.pi/2,90)
r_inner = 0.1
r_outer = r_inner + 0.1
x_inner, y_inner = polar2cats(r_inner, t_targ, x_targ, y_targ) 
x_outer, y_outer = polar2cats(r_outer, t_targ, x_targ, y_targ)


# SECTION - search for pose candidate
candidate = []
for i in range(len(x_inner)):

    x_ot = x_outer[i]
    y_ot = y_outer[i]

    a = [x_ot - x_targ, y_ot - y_targ]
    b = [1,0]
    dot = np.dot(a,b)
    alpha_candidate = np.arccos(dot/(np.linalg.norm(a)*np.linalg.norm(b)))
    desired_angle_joint = alpha_candidate - np.pi

    # from each approach candidate, create desired pose
    desired_pose = np.array([[x_targ], [y_targ], [desired_angle_joint]]) # input x, y, phi
    try:
        # try to solve for ik for joint config
        inv_solu = r.inverse_kinematic_geometry(desired_pose) # try to solve for inverse kinematic
    except:
        pass

    candidate.append(inv_solu)


# SECTION - plot task space
plt.axes().set_aspect('equal')
plt.axvline(x=0, c="black")
plt.axhline(y=0, c="black")
circle_plt(x_targ, y_targ, r_inner)
for i in range(len(x_inner)):
    plt.plot([x_inner[i],x_outer[i]],[y_inner[i],y_outer[i]],c ="orange")
for i in candidate:
    r.plot_arm(i)
plt.show()