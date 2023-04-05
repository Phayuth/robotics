import os
import sys
wd = os.path.abspath(os.getcwd())
sys.path.append(str(wd))

import numpy as np
import matplotlib.pyplot as plt
from rigid_body_transformation.homogeneous_transformation import inverse_hom_trans as invht
ax = plt.axes(projection='3d')

class ur5e:
    def __init__(self):
        # DH parameter is from https://www.researchgate.net/publication/347021253_Mathematical_Modelling_and_Simulation_of_Human-Robot_Collaboration
        # self.a     = np.array([      [0], [-0.425], [-0.3922],       [0],        [0],      [0]])
        # self.alpha = np.array([[np.pi/2],      [0],       [0], [np.pi/2], [-np.pi/2],      [0]])
        # self.d     = np.array([ [0.1625],      [0],       [0],  [0.1333],   [0.0997], [0.0996]])

        # DH parameter is from https://github.com/yorgoon/ur5_Final_Project/tree/master/inv_kin
        self.a     = np.array([      [0], [-0.425],  [-0.392],       [0],        [0],      [0]])
        self.alpha = np.array([[np.pi/2],      [0],       [0], [np.pi/2], [-np.pi/2],      [0]])
        self.d     = np.array([  [0.089],      [0],       [0],   [0.109],    [0.094],  [0.082]])

    def dh_transformation(self,theta,alpha,d,a):
        R = np.array([[np.cos(theta), -np.sin(theta)*np.cos(alpha),  np.sin(theta)*np.sin(alpha), a*np.cos(theta)],
                    [  np.sin(theta),  np.cos(theta)*np.cos(alpha), -np.cos(theta)*np.sin(alpha), a*np.sin(theta)],
                    [              0,                np.sin(alpha),                np.cos(alpha),               d],
                    [              0,                            0,                            0,               1]])
        return R

    def forward_kinematic(self, theta, return_full_H=False, return_each_H=False):

        A1 = self.dh_transformation(theta[0,0],self.alpha[0,0],self.d[0,0],self.a[0,0])
        A2 = self.dh_transformation(theta[1,0],self.alpha[1,0],self.d[1,0],self.a[1,0])
        A3 = self.dh_transformation(theta[2,0],self.alpha[2,0],self.d[2,0],self.a[2,0])
        A4 = self.dh_transformation(theta[3,0],self.alpha[3,0],self.d[3,0],self.a[3,0])
        A5 = self.dh_transformation(theta[4,0],self.alpha[4,0],self.d[4,0],self.a[4,0])
        A6 = self.dh_transformation(theta[5,0],self.alpha[5,0],self.d[5,0],self.a[5,0])

        T06 = A1 @ A2 @ A3 @ A4 @ A5 @ A6

        # x - y - z sequence
        # tan(roll) = r32/r33
        # tan(pitch)= -r31/(sqrt(r32^2 + r33^2))
        # tan(yaw)  = r21/r11
        # np.arctan2(y, x)
        
        roll = np.arctan2(T06[2,1],T06[2,2])
        pitch= np.arctan2(-T06[2,0],np.sqrt(T06[2,1]*T06[2,1] + T06[2,2]*T06[2,2]))
        yaw  = np.arctan2(T06[1,0],T06[0,0])

        x_current = np.array([[T06[0, 3]],
                              [T06[1, 3]],
                              [T06[2, 3]],
                              [     roll],
                              [    pitch],
                              [      yaw]])
        
        if return_full_H: # option to return transformation from base to end effector
            return T06
        if return_each_H: # option to return all transformation
            return A1, A2, A3, A4, A5, A6
        if not return_each_H and not return_each_H:
            return x_current
    

    def jacobian(self, theta):

        A1 = self.dh_transformation(theta[0,0],self.alpha[0,0],self.d[0,0],self.a[0,0])
        A2 = self.dh_transformation(theta[1,0],self.alpha[1,0],self.d[1,0],self.a[1,0])
        A3 = self.dh_transformation(theta[2,0],self.alpha[2,0],self.d[2,0],self.a[2,0])
        A4 = self.dh_transformation(theta[3,0],self.alpha[3,0],self.d[3,0],self.a[3,0])
        A5 = self.dh_transformation(theta[4,0],self.alpha[4,0],self.d[4,0],self.a[4,0])
        A6 = self.dh_transformation(theta[5,0],self.alpha[5,0],self.d[5,0],self.a[5,0])

        T01 = A1
        T02 = A1 @ A2
        T03 = A1 @ A2 @ A3
        T04 = A1 @ A2 @ A3 @ A4
        T05 = A1 @ A2 @ A3 @ A4 @ A5
        T06 = A1 @ A2 @ A3 @ A4 @ A5 @ A6

        Z0 = np.array([[0],
                       [0],
                       [1]])

        Z1 = np.array([[T01[0,2]],
                       [T01[1,2]],
                       [T01[2,2]]])

        Z2 = np.array([[T02[0,2]],
                       [T02[1,2]],
                       [T02[2,2]]])

        Z3 = np.array([[T03[0,2]],
                       [T03[1,2]],
                       [T03[2,2]]])

        Z4 = np.array([[T04[0,2]],
                       [T04[1,2]],
                       [T04[2,2]]])

        Z5 = np.array([[T05[0,2]],
                       [T05[1,2]],
                       [T05[2,2]]])

        O0 = np.array([[0],
                       [0],
                       [0]])

        O1 = np.array([[T01[3,0]],
                       [T01[3,1]],
                       [T01[3,2]]])

        O2 = np.array([[T02[3,0]],
                       [T02[3,1]],
                       [T02[3,2]]])

        O3 = np.array([[T03[3,0]],
                       [T03[3,1]],
                       [T03[3,2]]])

        O4 = np.array([[T04[3,0]],
                       [T04[3,1]],
                       [T04[3,2]]])

        O5 = np.array([[T05[3,0]],
                       [T05[3,1]],
                       [T05[3,2]]])

        O6 = np.array([[T06[3,0]],
                       [T06[3,1]],
                       [T06[3,2]]])

        Jv1 = np.transpose(np.cross(np.transpose(Z0),np.transpose(O6-O0))) 
        Jv2 = np.transpose(np.cross(np.transpose(Z1),np.transpose(O6-O1))) 
        Jv3 = np.transpose(np.cross(np.transpose(Z2),np.transpose(O6-O2))) 
        Jv4 = np.transpose(np.cross(np.transpose(Z3),np.transpose(O6-O3))) 
        Jv5 = np.transpose(np.cross(np.transpose(Z4),np.transpose(O6-O4))) 
        Jv6 = np.transpose(np.cross(np.transpose(Z5),np.transpose(O6-O5)))

        Jw1 = Z0
        Jw2 = Z1
        Jw3 = Z2
        Jw4 = Z3
        Jw5 = Z4
        Jw6 = Z5

        J1 = np.vstack((Jv1,Jw1))
        J2 = np.vstack((Jv2,Jw2))
        J3 = np.vstack((Jv3,Jw3))
        J4 = np.vstack((Jv4,Jw4))
        J5 = np.vstack((Jv5,Jw5))
        J6 = np.vstack((Jv6,Jw6))

        J = np.hstack((J1,J2,J3,J4,J5,J6))

        return J

    def jacobian_analytical(self, theta, roll, pitch, yaw):

        B = np.array([[np.cos(yaw)*np.sin(pitch), -np.sin(yaw), 0],
                      [np.sin(yaw)*np.sin(pitch),  np.cos(yaw), 0],
                      [            np.cos(pitch),            0, 1]])

        Binv = np.linalg.inv(B)

        Ja_mul = np.array([[1, 0, 0,         0,         0,         0],
                           [0, 1, 0,         0,         0,         0],
                           [0, 0, 1,         0,         0,         0],
                           [0, 0, 0, Binv[0,0], Binv[0,1], Binv[0,2]],
                           [0, 0, 0, Binv[1,0], Binv[1,1], Binv[1,2]],
                           [0, 0, 0, Binv[2,0], Binv[2,1], Binv[2,2]]])

        Ja = Ja_mul @ self.jacobian(theta)

        return Ja
    
    def inverse_kinematic_geometry(self, goal_desired):
        # given 4x4 transformation
        T06 = goal_desired

        d1 = self.d[0,0]
        d2 = self.d[1,0]
        d3 = self.d[2,0]
        d4 = self.d[3,0]
        d5 = self.d[4,0]
        d6 = self.d[5,0]

        a1 = self.a[0,0]
        a2 = self.a[1,0]
        a3 = self.a[2,0]
        a4 = self.a[3,0]
        a5 = self.a[4,0]
        a6 = self.a[5,0]

        alpha1 = self.alpha[0,0]
        alpha2 = self.alpha[1,0]
        alpha3 = self.alpha[2,0]
        alpha4 = self.alpha[3,0]
        alpha5 = self.alpha[4,0]
        alpha6 = self.alpha[5,0]

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
            T10 = invht(self.dh_transformation(theta[0,c], alpha1, d1, a1))
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
            T01 = self.dh_transformation(theta[0,c], alpha1, d1, a1)
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
            T01 = self.dh_transformation(theta[0,c], alpha1, d1, a1)
            T10 = invht(T01)
            T56 = self.dh_transformation(theta[5,c], alpha6, d6, a6)
            T65 = invht(T56)
            T45 = self.dh_transformation(theta[4,c], alpha5, d5, a5)
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
            T01 = self.dh_transformation(theta[0,c], alpha1, d1, a1)
            T10 = invht(T01)
            T56 = self.dh_transformation(theta[5,c], alpha6, d6, a6)
            T65 = invht(T56)
            T45 = self.dh_transformation(theta[4,c], alpha5, d5, a5)
            T54 = invht(T45)

            T14 = T10 @ T06 @ T65 @ T54
            p13 = T14 @ np.array([[0],[-d4],[0],[1]]) - np.array([[0],[0],[0],[1]])

            # theta 2 
            theta[1, c] = -np.arctan2(p13[1], -p13[0]) + np.arcsin(a3*np.sin(theta[2,c])/np.linalg.norm(p13))

            # theta 4
            T23 = self.dh_transformation(theta[2,c], alpha3, d3, a3)
            T32 = invht(T23)
            T12 = self.dh_transformation(theta[1,c], alpha2, d2, a2)
            T21 = invht(T12)
            T34 = T32 @ T21 @ T14
            theta[3, c] = np.arctan2(T34[1,0], T34[0,0])
        theta = theta.real

        return theta

    def plot_arm(self, theta, plt_basis=False, plt_show=False):

        A1 = self.dh_transformation(theta[0,0],self.alpha[0,0],self.d[0,0],self.a[0,0])
        A2 = self.dh_transformation(theta[1,0],self.alpha[1,0],self.d[1,0],self.a[1,0])
        A3 = self.dh_transformation(theta[2,0],self.alpha[2,0],self.d[2,0],self.a[2,0])
        A4 = self.dh_transformation(theta[3,0],self.alpha[3,0],self.d[3,0],self.a[3,0])
        A5 = self.dh_transformation(theta[4,0],self.alpha[4,0],self.d[4,0],self.a[4,0])
        A6 = self.dh_transformation(theta[5,0],self.alpha[5,0],self.d[5,0],self.a[5,0])

        T01 = A1
        T02 = A1 @ A2
        T03 = A1 @ A2 @ A3
        T04 = A1 @ A2 @ A3 @ A4
        T05 = A1 @ A2 @ A3 @ A4 @ A5
        T06 = A1 @ A2 @ A3 @ A4 @ A5 @ A6

        if plt_basis:
        # plot basic axis
            ax.plot3D([0, 0.5], [0, 0], [0, 0], 'red', linewidth=4)
            ax.plot3D([0, 0], [0, 0.5], [0, 0], 'purple', linewidth=4)
            ax.plot3D([0, 0], [0, 0], [0, 0.5], 'gray', linewidth=4)

        ax.plot3D([        0, T01[0, 3]], [        0, T01[1, 3]], [        0, T01[2, 3]], 'blue', linewidth=8)
        ax.plot3D([T01[0, 3], T02[0, 3]], [T01[1, 3], T02[1, 3]], [T01[2, 3], T02[2, 3]], 'orange', linewidth=8)
        ax.plot3D([T02[0, 3], T03[0, 3]], [T02[1, 3], T03[1, 3]], [T02[2, 3], T03[2, 3]], 'brown', linewidth=8)
        ax.plot3D([T03[0, 3], T04[0, 3]], [T03[1, 3], T04[1, 3]], [T03[2, 3], T04[2, 3]], 'pink', linewidth=8)
        ax.plot3D([T04[0, 3], T05[0, 3]], [T04[1, 3], T05[1, 3]], [T04[2, 3], T05[2, 3]], 'green', linewidth=8)
        ax.plot3D([T05[0, 3], T06[0, 3]], [T05[1, 3], T06[1, 3]], [T05[2, 3], T06[2, 3]], 'cyan', linewidth=8)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        if plt_show:
            plt.show()