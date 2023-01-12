import numpy as np

def dh_transformation(theta,alpha,d,a):
    R = np.array([[np.cos(theta), -np.sin(theta)*np.cos(alpha),  np.sin(theta)*np.sin(alpha), a*np.cos(theta)],
                  [np.sin(theta),  np.cos(theta)*np.cos(alpha), -np.cos(theta)*np.sin(alpha), a*np.sin(theta)],
                  [            0,                np.sin(alpha),                np.cos(alpha),               d],
                  [            0,                            0,                            0,               1]])
    return R

def hom_rotation_z_axis(theta):
    R = np.array([[np.cos(theta), -np.sin(theta),  0,    0],
                  [np.sin(theta),  np.cos(theta),  0,    0],
                  [            0,              0,  1,    0],
                  [            0,              0,  0,    1]])
    return R


# DH parameter is from this publication
# https://www.universal-robots.com/articles/ur/application-installation/dh-parameters-for-calculations-of-kinematics-and-dynamics/


theta1 = 0
theta2 = 0# -np.pi/2
theta3 = 0
theta4 = 0# np.pi/2
theta5 = 0
theta6 = 0

a1 = 0
a2 = -0.425
a3 = -0.3922
a4 = 0
a5 = 0
a6 = 0

alpha1 = np.pi/2
alpha2 = 0
alpha3 = 0
alpha4 = np.pi/2
alpha5 = -np.pi/2
alpha6 = 0

d1 = 0.1625
d2 = 0
d3 = 0
d4 = 0.1333
d5 = 0.0997
d6 = 0.0996

A1 = dh_transformation(theta1,alpha1,d1,a1)
A2 = dh_transformation(theta2,alpha2,d2,a2)
A3 = dh_transformation(theta3,alpha3,d3,a3)
A4 = dh_transformation(theta4,alpha4,d4,a4)
A5 = dh_transformation(theta5,alpha5,d5,a5)
A6 = dh_transformation(theta6,alpha6,d6,a6)

# correction = hom_rotation_z_axis(-np.pi/2)
T01 = A1
T02 = A1 @ A2
T03 = A1 @ A2 @ A3
T04 = A1 @ A2 @ A3 @ A4
T05 = A1 @ A2 @ A3 @ A4 @ A5
T06 = A1 @ A2 @ A3 @ A4 @ A5 @ A6

pe = np.array([[0],[0],[0],[1]]) # the fourth element MUST be equal to 1
p01 = T01 @ pe
p02 = T02 @ pe
p03 = T03 @ pe
p04 = T04 @ pe
p05 = T05 @ pe
p06 = T06 @ pe

print(p01)
print(p02)
print(p03)
print(p04)
print(p05)
print(p06)



import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(4,4))

ax = fig.add_subplot(111, projection='3d')

ax.scatter(0,0,0,label='origin')
ax.scatter(1,0,0,label='x')
ax.scatter(0,1,0,label='y')
ax.scatter(0,0,1,label='z')

ax.scatter(p01[0,0],p01[1,0],p01[2,0],label='j1')
ax.scatter(p02[0,0],p02[1,0],p02[2,0],label='j2')
ax.scatter(p03[0,0],p03[1,0],p03[2,0],label='j3')
ax.scatter(p04[0,0],p04[1,0],p04[2,0],label='j4')
ax.scatter(p05[0,0],p05[1,0],p05[2,0],label='j5')
ax.scatter(p06[0,0],p06[1,0],p06[2,0],label='j6')


ax.legend()
ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)
ax.set_zlim(-1, 1)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

plt.show()