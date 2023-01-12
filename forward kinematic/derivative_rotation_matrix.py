import numpy as np

def rotation_3d_x_axis(theta):
    R = np.array([[1,             0,              0],
                  [0, np.cos(theta), -np.sin(theta)],
                  [0, np.sin(theta),  np.cos(theta)]])
    return R

def rotation_3d_y_axis(theta):
    R = np.array([[np.cos(theta),  0,  np.sin(theta)],
                  [            0,  1,              0],
                  [-np.sin(theta), 0,  np.cos(theta)]])
    return R

def rotation_3d_z_axis(theta):
    R = np.array([[np.cos(theta), -np.sin(theta),  0],
                  [np.sin(theta),  np.cos(theta),  0],
                  [            0,              0,  1]])
    return R

def skew(x):
    return np.array([[      0,  -x[2,0],  x[1,0]],
                     [ x[2,0],        0, -x[0,0]],
                     [-x[1,0],   x[0,0],       0]])

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

# Derivative of rotation matrix about x = skew matrix of x basis vector @ Rotation Matrix x
theta = 0
R_dot_x = skew(x) @ rotation_3d_x_axis(theta)
print(R_dot_x)

# with omega, omega = (i vector basis)@theta_dot
theta_dot = 1
omega = theta_dot * x
R_dot_x = skew(omega) @ rotation_3d_x_axis(theta)
print(R_dot_x)