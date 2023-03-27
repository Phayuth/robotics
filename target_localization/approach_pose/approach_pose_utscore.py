import os
import sys
wd = os.path.abspath(os.getcwd())
sys.path.append(str(wd))

import numpy as np
import matplotlib.pyplot as plt
from robot.planar_rrr import planar_rrr


x_targ = 0.5 # user pick
y_targ = 0.5 # user pick

def polar2cats(r,theta):
    x = r*np.cos(theta) + x_targ
    y = r*np.sin(theta) + y_targ
    return x,y

# Create canidate pose
target_theta_list = np.linspace(np.pi/2,3*np.pi/2,20)

r_inner = 0.1
r_outer = r_inner + 0.1
x_inner, y_inner = polar2cats(r_inner,target_theta_list) 
x_outer, y_outer = polar2cats(r_outer,target_theta_list)


# create score of angle
score_angle = []


w1_angle = 0.5
for i in range(len(x_inner)):
    x_ot = x_outer[i]
    y_ot = y_outer[i]
    a = [x_ot - x_targ, y_ot - y_targ]
    b = [1,0]
    dot = np.dot(a,b)
    alpha_candidate = np.arccos(dot/(np.linalg.norm(a)*np.linalg.norm(b)))
    s_alpha = w1_angle*(1/(np.pi - alpha_candidate))
    score_angle.append(s_alpha)

w1_distant = 0.5
for i in range(len(x_inner)):
    x_ot = x_outer[i]
    y_ot = y_outer[i]
    a = [x_ot - x_targ, y_ot - y_targ]
    b = [1,0]
    dot = np.dot(a,b)
    alpha_candidate = np.arccos(dot/(np.linalg.norm(a)*np.linalg.norm(b)))
    s_alpha = w1_angle*(1/(np.pi - alpha_candidate))
    score_angle.append(s_alpha)



sc = np.array(score_angle)
print("==>> sc: \n", sc)




# find the maximum score
maxim = np.argmax(sc)
print("==>> maxim: \n", kmaxim)

# setup plot look
plt.axes().set_aspect('equal')
plt.axvline(x=0, c="black")
plt.axhline(y=0, c="black")

# plot data
plt.plot(x_inner,y_inner,c="green")

for i in range(len(x_inner)):
    plt.plot([x_inner[i],x_outer[i]],[y_inner[i],y_outer[i]],c ="orange")

plt.plot([x_inner[maxim],x_outer[maxim]],[y_inner[maxim],y_outer[maxim]],c ="cyan")

plt.show()
