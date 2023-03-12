import os
import sys
wd = os.path.abspath(os.getcwd())
sys.path.append(str(wd))

from robot.ur5e import ur5e
import numpy as np
from rigid_body_transformation.rotation_matrix import rotation_3d_x_axis, rotation_3d_y_axis, rotation_3d_z_axis

# Test
r = ur5e()
joints = np.array([[0],[0],[0],[0],[0],[0]])
joint_offset = np.array([[-np.pi/2],[-np.pi/2],[0],[-np.pi/2],[0],[0]])
joint_correction = joints - joint_offset
fktm = r.forward_kinematic(joint_correction, return_full_H=True) # T06

gs = np.array([ [ 0, -1, 0, 0.47],
                [ 0,  0, 1, 0.55],
                [-1,  0, 0, 0.12],
                [ 0,  0, 0,    1]]) # start pose # this is known by user

g_S_T = np.array([ [ 0, -1, 0, -0.3],
                [ 0,  0, 1, 0.39],
                [-1,  0, 0, 0.12],
                [ 0,  0, 0,    1]]) # goal pose # this is known by user


# bound joint angle -pi to pi
def bound(theta):
    if theta <= np.pi:
        theta = theta + 2*np.pi
    elif theta > np.pi:
        theta = theta - 2*np.pi
    return theta



# g_baseK_S = [ROTZ(-pi/2) [0 0 0.0892]'; 0 0 0 1]  %transformation from keating base to {S} # %-90 degree rotation around z and up x 0.0892 
g_baseK_S = np.array([[np.cos(-np.pi/2), -np.sin(-np.pi/2),  0,      0],
                    [  np.sin(-np.pi/2),  np.cos(-np.pi/2),  0,      0],
                    [                 0,                 0,  1, 0.0892],
                    [                 0,                 0,  0,      1]])


# g_T_toolK = [ROTX(-pi/2)*ROTY(pi/2) [0 0 0]'; 0 0 0 1] %transformation from {T} to keating tool  %-90 around x and 90 around y
rxry = rotation_3d_x_axis(-np.pi/2) @ rotation_3d_y_axis(np.pi/2)
temp1 = np.array([[0],[0],[0]])
temp2 = np.hstack((rxry,temp1))
temp3 = np.array([0,0,0,1])
g_T_toolK = np.vstack((temp2,temp3))

# transformation from keating base to keating tool 
# create desired tran matrix from base to ee and ee to tool
# g_des = g_baseK_S*g_S_T*g_T_toolK 
# g_des = g_baseK_S @ g_S_T @ g_T_toolK
g_des = g_S_T
# thetas = ur5InvKin(g_des)
theta_ik_result = r.inverse_kinematic_geo(g_des)


# current = g_baseK_S @ ur5FwdKin(thetas_unoffset) @ g_T_toolK
# thetas_unoffset = theta_ik_result + joint_offset
fknow = r.forward_kinematic(theta_ik_result, return_full_H=True)
# fknow = r.forward_kinematic(thetas_unoffset, return_full_H=True)
# currenttm = g_baseK_S @ fknow @ g_T_toolK

print("==>> current: ", fknow)

r.plot_arm(theta_ik_result,plt_basis=True, plt_show=True)
