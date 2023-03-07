import os
import sys
wd = os.path.abspath(os.getcwd())
sys.path.append(str(wd))

import numpy as np
import matplotlib.pyplot as plt
from robot.plannar_rrr import plannar_rrr

def polar2cats(r,theta):
    x = r*np.cos(theta)
    y = r*np.sin(theta)
    return x,y

# Create canidate pose
target_theta_list = np.linspace(np.pi/2,3*np.pi/2,90)
r_inner = 0.5
r_outer = 1
offset_from_origin_x = 5
offset_from_origin_y = 2

x_inner, y_inner = polar2cats(r_inner,target_theta_list) 
x_outer, y_outer = polar2cats(r_outer,target_theta_list)

x_inner, y_inner = x_inner + offset_from_origin_x, y_inner + offset_from_origin_y
x_outer, y_outer = x_outer + offset_from_origin_x, y_outer + offset_from_origin_y

# create robot instance
r = plannar_rrr.planar_rrr()

# search for pose candidate
candidate = []
for i in range(len(x_inner)):

    x_in = x_inner[i]
    y_in = y_inner[i]
    x_ot = x_outer[i]
    y_ot = y_outer[i]

    a = [x_ot - offset_from_origin_x, y_ot - offset_from_origin_y]
    b = [1,0]
    dot = np.dot(a,b)
    theta = np.arccos(dot/(np.linalg.norm(a)*np.linalg.norm(b)))
    theta_correction = theta - np.pi
    print("==>> theta_correction: ", theta_correction)

    # from desired pose, create desired pose
    desired_pose = np.array([[offset_from_origin_x], [offset_from_origin_y], [theta_correction]]) # input x, y, phi
    try:
        # try to solve for ik for joint config
        inv_solu = r.inverse_kinematic_geo(desired_pose) # only solve for robot at configuration elbow down
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