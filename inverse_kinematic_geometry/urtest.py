import os
import sys
wd = os.path.abspath(os.getcwd())
sys.path.append(str(wd))
from rigid_body_transformation.homogeneous_transformation import inverse_hom_trans as invht
import numpy as np
import matplotlib.pyplot as plt
from robot.ur5e import UR5e
np.set_printoptions(suppress=True)
robot = UR5e()
th = np.array([0,0,0,0,0,0]).reshape(6,1)
T06 = robot.forward_kinematic(th, return_full_H=True)
print("==>> T06: \n", T06)

a     = np.array([      [0], [-0.425],  [-0.392],       [0],        [0],      [0]])
alpha = np.array([[np.pi/2],      [0],       [0], [np.pi/2], [-np.pi/2],      [0]])
d     = np.array([  [0.089],      [0],       [0],   [0.109],    [0.094],  [0.082]])

def dh(theta,alpha,d,a):
    R = np.array([[np.cos(theta), -np.sin(theta)*np.cos(alpha),  np.sin(theta)*np.sin(alpha), a*np.cos(theta)],
                [  np.sin(theta),  np.cos(theta)*np.cos(alpha), -np.cos(theta)*np.sin(alpha), a*np.sin(theta)],
                [              0,                np.sin(alpha),                np.cos(alpha),               d],
                [              0,                            0,                            0,               1]])
    return R

d1 = d[0, 0]
d2 = d[1, 0]
d3 = d[2, 0]
d4 = d[3, 0]
d5 = d[4, 0]
d6 = d[5, 0]

a1 = a[0, 0]
a2 = a[1, 0]
a3 = a[2, 0]
a4 = a[3, 0]
a5 = a[4, 0]
a6 = a[5, 0]

alpha1 = alpha[0, 0]
alpha2 = alpha[1, 0]
alpha3 = alpha[2, 0]
alpha4 = alpha[3, 0]
alpha5 = alpha[4, 0]
alpha6 = alpha[5, 0]

theta = np.zeros((6, 8))

# theta 1
P05 = T06 @ np.array([0,0,-d6,1]).reshape(4,1) - np.array([0,0,0,1]).reshape(4,1)
p05x = P05[0, 0]
p05y = P05[1, 0]
p05xy = np.sqrt(p05x**2 + p05y**2)
if d4 > p05xy:
    print("no solution for th1")
psi = np.arctan2(p05y, p05x)
phi = np.arccos(d4 / p05xy)
theta1_1 = psi + phi + np.pi/2
theta1_2 = psi - phi + np.pi/2
theta[0, 0], theta[0, 1], theta[0, 2], theta[0, 3] = theta1_1, theta1_1, theta1_1, theta1_1
theta[0, 4], theta[0, 5], theta[0, 6], theta[0, 7] = theta1_2, theta1_2, theta1_2, theta1_2


# theta5
col = [0,1,4,5]
for i in range(8):
    p06x = T06[0, 3]
    p06y = T06[1, 3]
    theta5_1 =  np.arccos((p06x*np.sin(theta[0, i]) - p06y*np.cos(theta[0, i]) - d4)/d6)
    theta5_2 = -np.arccos((p06x*np.sin(theta[0, i]) - p06y*np.cos(theta[0, i]) - d4)/d6)
    if abs(p06x*np.sin(theta[0, i]) - p06y*np.cos(theta[0, i]) - d4) >= abs(d6):
        print("no solution for th5")
    if i in col:
        theta[4, i] = theta5_1
    else:
        theta[4, i] = theta5_2

# theta6
for i in range(8):
    T60 = invht(T06)
    Xy60 = T60[1,0]
    Yy60 = T60[1,1]
    Xx60 = T60[0,0]
    Yx60 = T60[0,1]
    if np.sin(theta[4, i]) == 0:
        print("theta6 solution is underdetermine in this case theta 6 can be random")
    theta6_term1 = (-Xy60*np.sin(theta[0, i]) + Yy60*np.cos(theta[0, i]))/(np.sin(theta[4, i]))
    theta6_term2 = (Xx60*np.sin(theta[0, i]) - Yx60*np.cos(theta[0, i]))/(np.sin(theta[4, i]))
    theta6 = np.arctan2(theta6_term1, theta6_term2)
    theta[5, i] = theta6



# theta3
col1 = [0,2,4,6]
col2 = [1,3,5,7]
for i in range(8):
    T10 = invht(dh(theta[0, i], alpha1, d1, a1))
    T65 = invht(dh(theta[5, i], alpha6, d6, a6))
    T54 = invht(dh(theta[4, i], alpha5, d5, a5))
    T14 = T10 @ T06 @ T65 @ T54
    p14x = T14[0,3]
    p14z = T14[2,3]

    p14xz = np.sqrt(p14x**2 + p14z**2)
    
    theta3_1 =  np.arccos((p14xz**2 - (a2**2) - (a3**2))/(2*a2*a3))
    theta3_2 = -np.arccos((p14xz**2 - (a2**2) - (a3**2))/(2*a2*a3))

    if i in col1:
        theta[2, i] = theta3_1
    else:
        theta[2, i] = theta3_2


# theta2
for i in range(8):
    T10 = invht(dh(theta[0, i], alpha1, d1, a1))
    T65 = invht(dh(theta[5, i], alpha6, d6, a6))
    T54 = invht(dh(theta[4, i], alpha5, d5, a5))
    T14 = T10 @ T06 @ T65 @ T54
    p14x = T14[0,3]
    p14z = T14[2,3]

    p14xz = np.sqrt(p14x**2 + p14z**2)
    theta2_term1 = np.arctan2(-p14z, -p14x)
    theta2_term2 = np.arcsin((-a3*np.sin(theta[2, i]))/ p14xz)
    theta2 = theta2_term1 - theta2_term2 

    theta[1, i] = theta2


# theta 4
for i in range(8):
    T23 = dh(theta[2, i], alpha3, d3, a3)
    T32 = invht(T23)
    T12 = dh(theta[1, i], alpha2, d2, a2)
    T21 = invht(T12)
    T34 = T32 @ T21 @ T14
    Xy34 = T34[1,0]
    Xx34 = T34[0,0]
    theta4 = np.arctan2(Xy34, Xx34)
    theta[3, i] = theta4





# print(theta.T)


# for j in range(8):
T_ik = robot.forward_kinematic(theta[:,0].reshape(6,1), return_full_H=True)
robot.plot_arm(theta[:,0].reshape(6,1),plt_basis=True, plt_show=True)
