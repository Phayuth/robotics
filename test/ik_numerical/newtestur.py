import numpy as np
from numpy import pi
np.set_printoptions(suppress=True)
import os
import sys
wd = os.path.abspath(os.getcwd())
sys.path.append(str(wd))
from rigid_body_transformation.homogeneous_transformation import inverse_hom_trans as invht

# a     = np.array([      [0], [-0.425],  [-0.392],       [0],        [0],      [0]])
# alpi = np.array([[np.pi/2],      [0],       [0], [np.pi/2], [-np.pi/2],      [0]])
# d     = np.array([  [0.089],      [0],       [0],   [0.109],    [0.094],  [0.082]])

alp0, alp1, alp2, alp3, alp4, alp5 = 0, pi/2, 0, 0, pi/2, -pi/2
a0, a1, a2, a3, a4, a5 =     0, 0, -0.425, -0.392,     0, 0
d1, d2, d3, d4, d5, d6 = 0.089, 0,      0,  0.109, 0.094, 0.082

def dhm(alpi, ai, dii, thii):
    R = np.array([[             np.cos(thii),             -np.sin(thii),              0,               ai],
                  [np.sin(thii)*np.cos(alpi), np.cos(thii)*np.cos(alpi), -np.sin(alpi), -np.sin(alpi)*dii],
                  [np.sin(thii)*np.sin(alpi), np.cos(thii)*np.sin(alpi),  np.cos(alpi),  np.cos(alpi)*dii],
                  [                          0,                           0,              0,            1]])
    return R


def forward(theta):
    T01 = dhm(alpi=alp0, ai=a0, dii=d1, thii=theta[0,0])
    T12 = dhm(alpi=alp1, ai=a1, dii=d2, thii=theta[1,0])
    T23 = dhm(alpi=alp2, ai=a2, dii=d3, thii=theta[2,0])
    T34 = dhm(alpi=alp3, ai=a3, dii=d4, thii=theta[3,0])
    T45 = dhm(alpi=alp4, ai=a4, dii=d5, thii=theta[4,0])
    T56 = dhm(alpi=alp5, ai=a5, dii=d6, thii=theta[5,0])

    T06 = T01 @ T12 @ T23 @ T34 @ T45 @ T56

    return T06


th_original = np.array([0.5,0.5,-0.5,0,0,0]).reshape(6,1)

T06 = forward(th_original)
print("==>> T06: \n", T06)

theta = np.zeros((6, 8))

# theta 1
P05 = T06 @ np.array([0,0,-d6,1]).reshape(4,1)
p05x = P05[0, 0]
p05y = P05[1, 0]
p05xy = np.sqrt((p05x**2) + (p05y**2))
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
    theta5 =  np.arccos((p06x*np.sin(theta[0, i]) - p06y*np.cos(theta[0, i]) - d4)/d6)

    if abs(p06x*np.sin(theta[0, i]) - p06y*np.cos(theta[0, i]) - d4) >= abs(d6):
        print("no solution for th5")

    if i in col:
        theta[4, i] = theta5
    else:
        theta[4, i] = -theta5



# theta6
for i in range(8):
    T60 = invht(T06)
    Xy60 = T60[1,0]
    Yy60 = T60[1,1]
    Xx60 = T60[0,0]
    Yx60 = T60[0,1]

    theta6_term1 = ((-Xy60*np.sin(theta[0, i])) + (Yy60*np.cos(theta[0, i])))/np.sin(theta[4, i])
    theta6_term2 = ( (Xx60*np.sin(theta[0, i])) - (Yx60*np.cos(theta[0, i])))/np.sin(theta[4, i])
    theta6 = np.arctan2(theta6_term1, theta6_term2)

    if np.sin(theta[4, i]) == 0:
        print("theta6 solution is underdetermine in this case theta 6 can be random")
        theta6 = 0

    theta[5, i] = theta6



# theta3
col1 = [0,2,4,6]
col2 = [1,3,5,7]
for i in range(8):
    T01 = dhm(alpi=alp0, ai=a0, dii=d1, thii=theta[0,i])
    T45 = dhm(alpi=alp4, ai=a4, dii=d5, thii=theta[4,i])
    T56 = dhm(alpi=alp5, ai=a5, dii=d6, thii=theta[5,i])

    T46 = T45 @ T56
    T60 = invht(T06)
    T40 = T46 @ T60
    T41 = T40 @ T01
    T14 = invht(T41)

    p14x = T14[0,3]
    p14z = T14[2,3]

    p14xz = np.sqrt(p14x**2 + p14z**2)
    
    theta3 =  np.arccos((p14xz**2 - (a2**2) - (a3**2))/(2*a2*a3))

    if i in col1:
        theta[2, i] = theta3
    else:
        theta[2, i] = -theta3


# theta2
for i in range(8):
    T01 = dhm(alpi=alp0, ai=a0, dii=d1, thii=theta[0,i])
    T45 = dhm(alpi=alp4, ai=a4, dii=d5, thii=theta[4,i])
    T56 = dhm(alpi=alp5, ai=a5, dii=d6, thii=theta[5,i])

    T46 = T45 @ T56
    T60 = invht(T06)
    T40 = T46 @ T60
    T41 = T40 @ T01
    T14 = invht(T41)

    p14x = T14[0,3]
    p14z = T14[2,3]

    p14xz = np.sqrt(p14x**2 + p14z**2)

    theta2_term1 = np.arctan2(-p14z, -p14x)
    theta2_term2 = np.arcsin((-a3*np.sin(theta[2, i]))/ p14xz)

    theta2 = theta2_term1 - theta2_term2 

    theta[1, i] = theta2




# theta 4
for i in range(8):
    T01 = dhm(alpi=alp0, ai=a0, dii=d1, thii=theta[0,i])
    T12 = dhm(alpi=alp1, ai=a1, dii=d2, thii=theta[1,i])
    T23 = dhm(alpi=alp2, ai=a2, dii=d3, thii=theta[2,i])
    T03 = T01 @ T12 @ T23

    T45 = dhm(alpi=alp4, ai=a4, dii=d5, thii=theta[4,i])
    T56 = dhm(alpi=alp5, ai=a5, dii=d6, thii=theta[5,i])
    T46 = T45 @ T56

    T30 = invht(T03)
    T64 = invht(T46)
    T34 = T30 @ T06 @ T64

    Xy34 = T34[1,0]
    Xx34 = T34[0,0]
    theta4 = np.arctan2(Xy34, Xx34)
    theta[3, i] = theta4












print(theta.T)





