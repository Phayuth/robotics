import os
import sys
wd = os.path.abspath(os.getcwd())
sys.path.append(str(wd))

import numpy as np
from numpy import pi
from rigid_body_transformation.homogeneous_transformation import inverse_hom_trans as invht
from rigid_body_transformation.dh_parameter import dh_transformation as dh
from robot.ur5e import ur5e
import matplotlib.pyplot as plt
from numpy import linalg


import cmath
import math
from math import cos as cos
from math import sin as sin
from math import atan2 as atan2
from math import acos as acos
from math import asin as asin
from math import sqrt as sqrt
from math import pi as pi

global mat
mat=np.matrix

# https://github.com/mc-capolei/python-Universal-robot-kinematics/blob/master/universal_robot_kinematics.py
# ****** Coefficients ******


global d1, a2, a3, a7, d4, d5, d6
d1 =  0.1273
a2 = -0.612
a3 = -0.5723
a7 = 0.075
d4 =  0.163941
d5 =  0.1157
d6 =  0.0922

global d, a, alph

# d = mat([0.1273, 0, 0, 0.163941, 0.1157, 0.0922])#ur10 mm
# a =mat([0 ,-0.612 ,-0.5723 ,0 ,0 ,0])#ur10 mm
# alph = mat([pi/2, 0, 0, pi/2, -pi/2, 0 ]) # ur10

d = mat([0.089159, 0, 0, 0.10915, 0.09465, 0.0823]) #ur5
a = mat([0 ,-0.425 ,-0.39225 ,0 ,0 ,0]) #ur5
alph = mat([math.pi/2, 0, 0, math.pi/2, -math.pi/2, 0 ])  #ur5



# ************************************************** FORWARD KINEMATICS

def AH( n,th,c  ):

  T_a = mat(np.identity(4), copy=False)
  T_a[0,3] = a[0,n-1]
  T_d = mat(np.identity(4), copy=False)
  T_d[2,3] = d[0,n-1]

  Rzt = mat([[cos(th[n-1,c]), -sin(th[n-1,c]), 0 ,0],
	         [sin(th[n-1,c]),  cos(th[n-1,c]), 0, 0],
	         [0,               0,              1, 0],
	         [0,               0,              0, 1]],copy=False)
      

  Rxa = mat([[1, 0,                 0,                  0],
			 [0, cos(alph[0,n-1]), -sin(alph[0,n-1]),   0],
			 [0, sin(alph[0,n-1]),  cos(alph[0,n-1]),   0],
			 [0, 0,                 0,                  1]],copy=False)

  A_i = T_d * Rzt * T_a * Rxa
	    

  return A_i

def HTrans(th,c ):  
  A_1=AH( 1,th,c  )
  A_2=AH( 2,th,c  )
  A_3=AH( 3,th,c  )
  A_4=AH( 4,th,c  )
  A_5=AH( 5,th,c  )
  A_6=AH( 6,th,c  )
      
  T_06=A_1*A_2*A_3*A_4*A_5*A_6

  return T_06

# ************************************************** INVERSE KINEMATICS 

def invKine(desired_pos):# T60
    th = mat(np.zeros((6, 8)))
    P_05 = (desired_pos * mat([0,0, -d6, 1]).T-mat([0,0,0,1 ]).T)
    
    # **** theta1 ****
    
    psi = atan2(P_05[2-1,0], P_05[1-1,0])
    phi = acos(d4 /sqrt(P_05[2-1,0]*P_05[2-1,0] + P_05[1-1,0]*P_05[1-1,0]))
    #The two solutions for theta1 correspond to the shoulder
    #being either left or right
    th[0, 0:4] = pi/2 + psi + phi
    th[0, 4:8] = pi/2 + psi - phi
    th = th.real
    
    # **** theta5 ****
    
    cl = [0, 4]# wrist up or down
    for i in range(0,len(cl)):
            c = cl[i]
            T_10 = linalg.inv(AH(1,th,c))
            T_16 = T_10 * desired_pos
            th[4, c:c+2] = + acos((T_16[2,3]-d4)/d6);
            th[4, c+2:c+4] = - acos((T_16[2,3]-d4)/d6);

    th = th.real
  
    # **** theta6 ****
    # theta6 is not well-defined when sin(theta5) = 0 or when T16(1,3), T16(2,3) = 0.

    cl = [0, 2, 4, 6]
    for i in range(0,len(cl)):
            c = cl[i]
            T_10 = linalg.inv(AH(1,th,c))
            T_16 = linalg.inv( T_10 * desired_pos )
            th[5, c:c+2] = atan2((-T_16[1,2]/sin(th[4, c])),(T_16[0,2]/sin(th[4, c])))
            
    th = th.real

    # **** theta3 ****
    cl = [0, 2, 4, 6]
    for i in range(0,len(cl)):
            c = cl[i]
            T_10 = linalg.inv(AH(1,th,c))
            T_65 = AH( 6,th,c)
            T_54 = AH( 5,th,c)
            T_14 = ( T_10 * desired_pos) * linalg.inv(T_54 * T_65)
            P_13 = T_14 * mat([0, -d4, 0, 1]).T - mat([0,0,0,1]).T
            t3 = cmath.acos((linalg.norm(P_13)**2 - a2**2 - a3**2 )/(2 * a2 * a3)) # norm ?
            th[2, c] = t3.real
            th[2, c+1] = -t3.real

    # **** theta2 and theta 4 ****

    cl = [0, 1, 2, 3, 4, 5, 6, 7]
    for i in range(0,len(cl)):
            c = cl[i]
            T_10 = linalg.inv(AH( 1,th,c ))
            T_65 = linalg.inv(AH( 6,th,c))
            T_54 = linalg.inv(AH( 5,th,c))
            T_14 = (T_10 * desired_pos) * T_65 * T_54
            P_13 = T_14 * mat([0, -d4, 0, 1]).T - mat([0,0,0,1]).T
            
            # theta 2
            th[1, c] = -atan2(P_13[1], -P_13[0]) + asin(a3* sin(th[2,c])/linalg.norm(P_13))
            # theta 4
            T_32 = linalg.inv(AH( 3,th,c))
            T_21 = linalg.inv(AH( 2,th,c))
            T_34 = T_32 * T_21 * T_14
            th[3, c] = atan2(T_34[1,0], T_34[0,0])
    th = th.real

    return th



gs = np.array([[ 0, -1, 0, 0.47],
               [ 0,  0, 1, 0.55],
               [-1,  0, 0, 0.12],
               [ 0,  0, 0,    1]])

thetaette = invKine(gs)
# print("==>> thetaette: \n", thetaette)

index_solution = 2
T06 = HTrans(thetaette,index_solution)
print("==>> T06: \n", T06)

rob = ur5e()
ttt = rob.forward_kinematic(thetaette[:,index_solution], return_full_H=True)
print("==>> ttt: \n", ttt)





















# -----------------------------------------------------------------------------------------------------
import os
import sys
wd = os.path.abspath(os.getcwd())
sys.path.append(str(wd))

import numpy as np
from numpy import pi
from rigid_body_transformation.dh_parameter import dh_transformation as dh
from rigid_body_transformation.homogeneous_transformation import inverse_hom_trans as invht
# https://github.com/mc-capolei/python-Universal-robot-kinematics/blob/master/universal_robot_kinematics.py

np.set_printoptions(threshold=100)
d1 = 0.0892
d2 = 0
d3 = 0
d4 = 0.1093
d5 = 0.09475
d6 = 0.0825
a1 = 0
a2 = -0.425
a3 = -0.392
a4 = 0
a5 = 0
a6 = 0
alpha1 = pi/2
alpha2 = 0
alpha3 = 0
alpha4 = pi/2
alpha5 = -pi/2
alpha6 = 0

T06 = np.array([[ 0, -1, 0, 0.47],
               [ 0,  0, 1, 0.55],
               [-1,  0, 0, 0.12],
               [ 0,  0, 0,    1]])

theta = np.zeros((6,8))
P05 = T06 @ np.array([[0],[0],[-d6],[1]]) - np.array([[0],[0],[0],[1]])

# theta 1
p05y = P05[1,0]
p05x = P05[0,0]
psi = np.arctan2(p05y, p05x)
phi = np.arccos(d4 / np.sqrt(p05x**2 + p05y**2))
#The two solutions for theta1 correspond to the shoulder being either left or right
theta[0, 0:4] = np.pi/2 + psi + phi
theta[0, 4:8] = np.pi/2 + psi - phi
theta = theta.real


# theta 5 # wrist up or down
cl = [0, 4]
for i in range(0,len(cl)):
    c = cl[i]
    T10 = invht(dh(theta[0,c], alpha1, d1, a1))
    T16 = T10 @ T06
    p16z = T16[2,3]
    theta[4, c:c+2]   =   np.arccos((p16z-d4)/d6)
    theta[4, c+2:c+4] = - np.arccos((p16z-d4)/d6)
theta = theta.real

# theta 6
# theta6 is not well-defined when sin(theta5) = 0 or when T16(1,3), T16(2,3) = 0.

cl = [0, 2, 4, 6]
for i in range(0,len(cl)):
    c = cl[i]
    T01 = dh(theta[0,c], alpha1, d1, a1)
    T10 = invht(T01)
    T16 = T06 @ T10
    zy = T16[1,2]
    zx = T16[0,2]
    term1 = -zy/np.sin(theta[4,c])
    term2 =  zx/np.sin(theta[4,c])
    theta[5, c:c+2] = np.arctan2(term1, term2)
theta = theta.real

# theta 3
cl = [0, 2, 4, 6]
for i in range(0,len(cl)):
    c = cl[i]
    T01 = dh(theta[0,c], alpha1, d1, a1)
    T10 = invht(T01)
    T56 = dh(theta[5,c], alpha6, d6, a6)
    T65 = invht(T56)
    T45 = dh(theta[4,c], alpha5, d5, a5)
    T54 = invht(T45)

    T14 = T10 @ T06 @ T65 @ T54

    p13 = T14 @ np.array([[0],[-d4],[0],[1]]) - np.array([[0],[0],[0],[1]])
    
    term = (np.linalg.norm(p13)**2 - a2**2 - a3**2)/(2*a2*a3)
    theta[2, c] = np.arccos(term)
    theta[2, c+1] = -np.arccos(term)
theta = theta.real


#theta 2 and theta 4
cl = [0, 1, 2, 3, 4, 5, 6, 7]
for i in range(0,len(cl)):
    c = cl[i]
    T01 = dh(theta[0,c], alpha1, d1, a1)
    T10 = invht(T01)
    T56 = dh(theta[5,c], alpha6, d6, a6)
    T65 = invht(T56)
    T45 = dh(theta[4,c], alpha5, d5, a5)
    T54 = invht(T45)

    T14 = T10 @ T06 @ T65 @ T54
    p13 = T14 @ np.array([[0],[-d4],[0],[1]]) - np.array([[0],[0],[0],[1]])

    # theta2 
    theta[1, c] = -np.arctan2(p13[1], -p13[0]) + np.arcsin(a3*np.sin(theta[2,c])/np.linalg.norm(p13))

    # theta 4
    T23 = dh(theta[2,c], alpha3, d3, a3)
    T32 = invht(T23)
    T12 = dh(theta[1,c], alpha2, d2, a2)
    T21 = invht(T12)
    T34 = T32 @ T21 @ T14
    theta[3, c] = np.arctan2(T34[1,0], T34[0,0])
theta = theta.real


print(theta)

















# ------------------------------------------------------------------------
# % Inverse Kinematics Test
# rosshutdown;
# ur5 = ur5_interface();
# joint_offset = [-pi/2 -pi/2 0 -pi/2 0 0]';
# joints = [0 0 0 0 0 0]';
# g_S_T = ur5FwdKin(joints- joint_offset);
# g_baseK_S = [ROTZ(-pi/2) [0 0 0.0892]'; 0 0 0 1];  %transformation from keating base to {S}
# %-90 degree rotation around z and up x 0.0892 
# baseKFrame = tf_frame('S','base_K',eye(4));
# baseKFrame.move_frame('S',inv(g_baseK_S));

# g_T_toolK = [ROTX(-pi/2)*ROTY(pi/2) [0 0 0]'; 0 0 0 1]; %transformation from {T} to keating tool 
# %-90 around x and 90 around y
# toolKFrame = tf_frame('T','tool_K',eye(4));
# toolKFrame.move_frame('T',g_T_toolK);

# g_des = g_baseK_S*g_S_T*g_T_toolK; %transformation from keating base to keating tool 
# thetas = ur5InvKin(g_des);

# function [ theta ] = ur5InvKin( gd )
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# %%%%%%%% UR5INV - Long Qian   %%%%%%%%%%%%%%%%%
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# %   Inverse kinematics of UR5
# %   Formulas adapted from Ryan Keating's "UR5 Inverse Kinematics" paper
# %   Input args:
# %           gd - the 4x4 matrix - desired transformation from base to end
# %   Output args:
# %           theta - 6x8 matrix, each colum represents one possible solution 
# %           of joint angles

#     % Check input size
#     if ~isequal(size(gd), [4,4])
#         msg ='ur5invKin function: Wrong input format, must be a 4x4 matrix';
#         error(msg);
#     end

#     theta = zeros(6, 8);
#     % DH parameters
#     d1 = 0.0892;
#     d2 = 0;
#     d3 = 0;
#     d4 = 0.1093;
#     d5 = 0.09475;
#     d6 = 0.0825;
#     a1 = 0;
#     a2 = -0.425;
#     a3 = -0.392;
#     a4 = 0;
#     a5 = 0;
#     a6 = 0;
#     alpha1 = pi/2;
#     alpha2 = 0;
#     alpha3 = 0;
#     alpha4 = pi/2;
#     alpha5 = -pi/2;
#     alpha6 = 0;
    
#     % Calculating theta1
#     p05 = gd * [0, 0, -d6, 1]'  - [0, 0, 0, 1]';
#     psi = atan2(p05(2), p05(1));
#     phi = acos(d4 / sqrt(p05(2)*p05(2) + p05(1)*p05(1)));
#     theta(1, 1:4) = pi/2 + psi + phi;
#     theta(1, 5:8) = pi/2 + psi - phi;
#     theta = real(theta);
    
    
#     % Calculating theta5
#     cols = [1, 5];
#     for i=1:length(cols)
#         c = cols(i);
#         T10 = inv(DH(a1, alpha1, d1, theta(1,c)));
#         T16 = T10 * gd;
#         p16z = T16(3,4);
#         t5 = acos((p16z-d4)/d6);
#         theta(5, c:c+1) = t5;
#         theta(5, c+2:c+3) = -t5;
#     end
#     theta = real(theta);
    
#     % Calculating theta6
#     cols = [1, 3, 5, 7];
#     for i=1:length(cols)
#         c = cols(i);
#         T01 = DH(a1, alpha1, d1, theta(1,c));
#         T61 = inv(gd) * T01;
#         T61zy = T61(2, 3);
#         T61zx = T61(1, 3);
#         t5 = theta(5, c);
#         theta(6, c:c+1) = atan2(-T61zy/sin(t5), T61zx/sin(t5));
#     end
#     theta = real(theta);
    
#     % Calculating theta3
#     cols = [1, 3, 5, 7];
#     for i=1:length(cols)
#         c = cols(i);
#         T10 = inv(DH(a1, alpha1, d1, theta(1,c)));
#         T65 = inv(DH(a6, alpha6, d6, theta(6,c)));
#         T54 = inv(DH(a5, alpha5, d5, theta(5,c)));
#         T14 = T10 * gd * T65 * T54;
#         p13 = T14 * [0, -d4, 0, 1]' - [0,0,0,1]';
#         p13norm2 = norm(p13) ^ 2;
#         t3p = acos((p13norm2-a2*a2-a3*a3)/(2*a2*a3));
#         theta(3, c) = t3p;
#         theta(3, c+1) = -t3p;
#     end
#     theta = real(theta);
    
#     % Calculating theta2 and theta4
#     cols = [1, 2, 3, 4, 5, 6, 7, 8];
#     for i=1:length(cols)
#         c = cols(i);
#         T10 = inv(DH(a1, alpha1, d1, theta(1,c)));
#         T65 = inv(DH(a6, alpha6, d6, theta(6,c)));
#         T54 = inv(DH(a5, alpha5, d5, theta(5,c)));
#         T14 = T10 * gd * T65 * T54;
#         p13 = T14 * [0, -d4, 0, 1]' - [0,0,0,1]';
#         p13norm = norm(p13);
#         theta(2, c) = -atan2(p13(2), -p13(1))+asin(a3*sin(theta(3,c))/p13norm);
#         T32 = inv(DH(a3, alpha3, d3, theta(3,c)));
#         T21 = inv(DH(a2, alpha2, d2, theta(2,c)));
#         T34 = T32 * T21 * T14;
#         theta(4, c) = atan2(T34(2,1), T34(1,1));
#     end
#     theta = real(theta);
    
#     for j=1:8
#         theta(1,j) = theta(1,j)-pi;
#     end
    
#     % Bound the joint angles from -pi to pi
#     for i=1:6
#         for j=1:8
#             if theta(i,j) <= -pi
#                 theta(i,j) = theta(i,j) + 2*pi;
#             elseif theta(i,j) > pi
#                 theta(i,j) = theta(i,j) - 2*pi;
#             end
#         end
#     end
                
                    
# end
