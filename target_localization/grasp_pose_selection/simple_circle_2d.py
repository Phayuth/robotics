import os
import sys
wd = os.path.abspath(os.getcwd())
sys.path.append(str(wd))

import numpy as np
import matplotlib.pyplot as plt
from robot.planar_rrr import planar_rrr

def polar2cats(r,theta):
    x = r*np.cos(theta)
    y = r*np.sin(theta)
    return x,y

# Create canidate pose
target_theta_list = np.linspace(np.pi/2,3*np.pi/2,90)
# target_theta_list = np.linspace(0,2*np.pi,180)
r_inner = 0.1
r_outer = r_inner + 0.1
offset_from_origin_x = 1.5
offset_from_origin_y = 1.5

x_inner, y_inner = polar2cats(r_inner,target_theta_list) 
x_outer, y_outer = polar2cats(r_outer,target_theta_list)

x_inner = x_inner + offset_from_origin_x 
y_inner = y_inner + offset_from_origin_y
x_outer = x_outer + offset_from_origin_x
y_outer = y_outer + offset_from_origin_y

# create robot instance
r = planar_rrr()

# search for pose candidate
candidate = []
for i in range(len(x_inner)):

    x_ot = x_outer[i]
    y_ot = y_outer[i]

    a = [x_ot - offset_from_origin_x, y_ot - offset_from_origin_y]
    b = [1,0]
    dot = np.dot(a,b)
    alpha_candidate = np.arccos(dot/(np.linalg.norm(a)*np.linalg.norm(b)))
    desired_angle_joint = alpha_candidate - np.pi
    print("==>> desired_angle_joint: ", desired_angle_joint)

    # from each approach candidate, create desired pose
    desired_pose = np.array([[offset_from_origin_x], [offset_from_origin_y], [desired_angle_joint]]) # input x, y, phi
    try:
        # try to solve for ik for joint config
        inv_solu = r.inverse_kinematic_geometry(desired_pose) # try to solve for inverse kinematic
    except:
        pass

    candidate.append(inv_solu)

# setup plot look
plt.axes().set_aspect('equal')
plt.axvline(x=0, c="black")
plt.axhline(y=0, c="black")

# plot data
plt.plot(x_inner,y_inner,c="green")

for i in range(len(x_inner)):
    plt.plot([x_inner[i],x_outer[i]],[y_inner[i],y_outer[i]],c ="orange")
for i in candidate:
    r.plot_arm(i)

plt.show()