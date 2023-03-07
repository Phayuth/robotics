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

target_theta_list = np.linspace(np.pi/2,3*np.pi/2,90)
r_inner = 0.5
r_outer = 1
offset_from_origin_x = 5
offset_from_origin_y = 2

x_inner, y_inner = polar2cats(r_inner,target_theta_list) 
x_outer, y_outer = polar2cats(r_outer,target_theta_list)

x_inner, y_inner = x_inner + offset_from_origin_x, y_inner + offset_from_origin_y
x_outer, y_outer = x_outer + offset_from_origin_x, y_outer + offset_from_origin_y

r = plannar_rrr.planar_rrr()

# Bruteforce find config angle that match paralell to any of grasp pose candidate
result = []
theta = np.linspace(0,2*np.pi,360)

for i in theta:
    for j in theta:
        for k in theta:

            a = np.array([[i], [j], [k]])
            link = r.forward_kinematic(a)

            for i in range(len(x_inner)):

                x_in = x_inner[i]
                y_in = y_inner[i]
                x_ot = x_outer[i]
                y_ot = y_outer[i]
        
                a = [x_ot - x_in, y_ot - y_in]
                b = [link[3,0] - link[2,0], link[3,1] - link[2,1]]
                cross = np.cross(a,b)

                if cross == 0:
                    print(i,j,k)
                    result.append([i,j,k])
                else:
                    print(i,j,k,"Not this one")

print(result)
