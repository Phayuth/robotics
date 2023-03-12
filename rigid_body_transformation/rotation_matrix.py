import numpy as np

# 2d rotation
def rotation_2d(theta):
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta),  np.cos(theta)]])
    return R

# 3d rotation
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
