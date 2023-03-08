from ur5e import ur5e
from planar_rr import planar_rr
from planar_rrr import planar_rrr
from spatial_rr import spatial_rr
from spatial_rrp import spatial_rrp
import numpy as np
from rigid_body_transformation import rotation_matrix

# Test
r = ur5e()

xx = 0
yx = 0
zx = -1

xy = 0
yy = -1
zy = 0

xz = -1
yz = 0
zz = 0

px = 0.8173
py = 0.1914
pz = 0.0837

gd = np.array([[xx, yx, zx, px],
                [xy, yy, zy, py],
                [xz, yz, zz, pz],
                [ 0,  0,  0,  1]]) # this is known by user


# bound joint angle -pi to pi
def bound(theta):
    if theta <= np.pi:
        theta = theta + 2*np.pi
    elif theta > np.pi:
        theta = theta - 2*np.pi
    return theta

# current = r.forward_kinematic(a)
# theta_ik = r.inverse_kinematic_geo(gd)
# current = r.forward_kinematic(theta_ik)

# r.plot_arm(theta_ik)

joint_offset = np.array([[-np.pi/2],[-np.pi/2],[0],[-np.pi/2],[0],[0]])
joints = np.array([[0],[0],[0],[0],[0],[0]])
joint_correction = joints - joint_offset
_, g_S_T = r.forward_kinematic(joints)
r.plot_arm(joints)

# g_baseK_S = [ROTZ(-pi/2) [0 0 0.0892]'; 0 0 0 1]  %transformation from keating base to {S} # %-90 degree rotation around z and up x 0.0892 
g_baseK_S = np.array([[np.cos(-np.pi/2), -np.sin(-np.pi/2),  0,      0],
                      [np.sin(-np.pi/2),  np.cos(-np.pi/2),  0,      0],
                      [               0,                 0,  1, 0.0892],
                      [               0,                 0,  0,      1]])


# g_T_toolK = [ROTX(-pi/2)*ROTY(pi/2) [0 0 0]'; 0 0 0 1] %transformation from {T} to keating tool  %-90 around x and 90 around y
rxry = rotation_matrix.rotation_3d_x_axis(-np.pi/2) @ rotation_matrix.rotation_3d_y_axis(np.pi/2)
temp1 = np.array([[0],[0],[0]])
temp2 = np.hstack((rxry,temp1))
temp3 = np.array([0,0,0,1])
g_T_toolK = np.vstack((temp2,temp3))

# g_des = g_baseK_S*g_S_T*g_T_toolK %transformation from keating base to keating tool 
g_des = g_baseK_S @ g_S_T @ g_T_toolK
# thetas = ur5InvKin(g_des)
theta_ik_result = r.inverse_kinematic_geo(g_des)

# thetas_unoffset = theta_ik_result + joint_offset
# forward_kineee = g_baseK_S @ ur5FwdKin(thetas_unoffset) @ g_T_toolK
