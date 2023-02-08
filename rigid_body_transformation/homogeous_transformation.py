import numpy as np

def hom_rotation_x_axis(theta):
    R = np.array([[1,             0,              0,    0],
                  [0, np.cos(theta), -np.sin(theta),    0],
                  [0, np.sin(theta),  np.cos(theta),    0],
                  [0,             0,              0,    1]])
    return R

def hom_rotation_y_axis(theta):
    R = np.array([[np.cos(theta),  0,  np.sin(theta),    0],
                  [            0,  1,              0,    0],
                  [-np.sin(theta), 0,  np.cos(theta),    0],
                  [0,             0,               0,    1]])
    return R

def hom_rotation_z_axis(theta):
    R = np.array([[np.cos(theta), -np.sin(theta),  0,    0],
                  [np.sin(theta),  np.cos(theta),  0,    0],
                  [            0,              0,  1,    0],
                  [            0,              0,  0,    1]])
    return R

def hom_pure_translation(x,y,z):
    R = np.array([[            1,              0,  0,    x],
                  [            0,              1,  0,    y],
                  [            0,              0,  1,    z],
                  [            0,              0,  0,    1]])
    return R
    