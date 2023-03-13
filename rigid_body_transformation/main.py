import numpy as np
from homogeneous_transformation import *
from rotation_matrix import *
from plot_frame import *


# homog transform -----------------------------------------------------------------------------------------------------------------------------
# print("This is the Homogeneous Transforamtion Test")
theta = np.deg2rad(40)
p1 = np.array([[1],[1],[1],[1]]) # the fourth element MUST be equal to 1

H01 = hom_pure_translation(5,0,0)
H12 = hom_pure_translation(0,5,0)
H23 = hom_pure_translation(0,0,5)

H = H01@H12@H23
p0 = H@p1

inv_tran = inverse_hom_trans(H01)
print("==>> inv_tran: \n", inv_tran)
print("==>> H01: \n", H01)
#-------------------------------------------------------------------------------------------------------------------------------------------





# rotation------------------------------------------------------------------------------------------------------------------------------
# print("This is the ratation matrix test")
# 2d
theta = np.deg2rad(90)
p1 = np.array([[1],[0]])
p2 = rotation_2d(theta) @ p1
# print("This is 2d rotation")
# print(p1)
# print(p2)

# 3d
theta = np.deg2rad(40)
p1 = np.array([[1],[0],[0]])
p2 = rotation_3d_y_axis(theta) @ p1
# print("This is 3d rotation")
# print(p1)
# print(p2)

# sequence of rotation
# for concurrent frame rotation, we post multiply of rotation matrix
theta = np.deg2rad(90)
p1 = np.array([[1],[0],[0]])
p2 = rotation_3d_y_axis(theta) @ rotation_3d_z_axis(theta) @ p1
# print("This concurrent frame rotation")
# print(p1)
# print(p2)

# for fixed frame rotation, we pre multiply of rotation matrix
theta = np.deg2rad(90)
p1 = np.array([[1],[0],[0]])
p2 = rotation_3d_z_axis(theta) @ rotation_3d_y_axis(theta) @ p1
# print("This is fixed frame rotation")
# print(p1)
# print(p2)
#-------------------------------------------------------------------------------------------------------------------------------------------





# plot frame -------------------------------------------------------------------------------------------------------------------------------
theta = np.pi/4

# rot = rotation_2d(theta)
# tran = np.array([[1],[1]])
# plot_frame_2d(rot,tran)

g = np.array([[np.cos(theta), -np.sin(theta),  0,    1],
                [np.sin(theta),  np.cos(theta),  0,    1],
                [            0,              0,  1,    1],
                [            0,              0,  0,    1]])

plot_frame_3d(g,plt_basis=True,plt_show=True)
#-------------------------------------------------------------------------------------------------------------------------------------------
