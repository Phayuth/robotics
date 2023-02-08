import numpy as np

def dh_transformation(theta,alpha,d,a):
    R = np.array([[np.cos(theta), -np.sin(theta)*np.cos(alpha),  np.sin(theta)*np.sin(alpha), a*np.cos(theta)],
                  [np.sin(theta),  np.cos(theta)*np.cos(alpha), -np.cos(theta)*np.sin(alpha), a*np.sin(theta)],
                  [            0,                np.sin(alpha),                np.cos(alpha),               d],
                  [            0,                            0,                            0,               1]])
    return R



# good old 2d rr joint
theta1 = 0
theta2 = 0
a1 = 1
a2 = 1
alpha1 = 0
alpha2 = 0
d1 = 0
d2 = 0

A1 = dh_transformation(theta1,alpha1,d1,a1)
A2 = dh_transformation(theta2,alpha2,d2,a2)

T = A1 @ A2
p2 = np.array([[0],[0],[0],[1]]) # the fourth element MUST be equal to 1
p0 = T@p2

# print(p0)


# 3d 2 link 2 revolute joint
theta1 = 0
theta2 = 0
a1 = 0
a2 = 1
alpha1 = np.pi/2
alpha2 = 0
d1 = 1
d2 = 0

A1 = dh_transformation(theta1,alpha1,d1,a1)
A2 = dh_transformation(theta2,alpha2,d2,a2)

T = A1 @ A2
p2 = np.array([[0],[0],[0],[1]]) # the fourth element MUST be equal to 1
p0 = T@p2

print(p0)