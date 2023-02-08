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

J1 = np.append(Jv1,Jw1,axis=0)
J2 = np.append(Jv2,Jw2,axis=0)
J3 = np.append(Jv3,Jw3,axis=0)
J4 = np.append(Jv4,Jw4,axis=0)
J5 = np.append(Jv5,Jw5,axis=0)
J6 = np.append(Jv6,Jw6,axis=0)


print(J1)
print(J2)
print(J3)
print(J4)
print(J5)
print(J6)

print("Jac")
J = np.append(np.append(np.append(np.append(np.append(J1,J2,axis=1),J3,axis=1),J4,axis=1),J5,axis=1),J6,axis=1)
print(J)
print(np.shape(J))