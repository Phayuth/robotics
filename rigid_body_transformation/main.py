import numpy as np
from homogeneous_transformation import *
from quaternion import *
from rotation_matrix import *
from skew_matrix import *
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

# print(H)
# print(p1)
# print(p0)
#-------------------------------------------------------------------------------------------------------------------------------------------






# quaternion -----------------------------------------------------------------------------------------------------------------------------
# print("This is the quaternion test")
unit_vector_x = np.array([1,0,0])
unit_vector_y = np.array([0,1,0])
unit_vector_z = np.array([0,0,1])

theta = np.deg2rad(90)
# print(theta)

q = quaternion_from_theta_and_vector(theta,unit_vector_x)
theta_ori,n_ori = quaternion_to_theta_and_vector(q)

# print(q)
# print(theta_ori,n_ori)
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






# skew matrix ------------------------------------------------------------------------------------------------------------------------------
# print("This is the skew matrix test")
# Vector basis
x = np.array([[1],
              [0],
              [0]])

y = np.array([[0],
              [1],
              [0]])

z = np.array([[0],
              [0],
              [1]])

# print(skew(x))
# print(skew(y))
# print(skew(z))
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
