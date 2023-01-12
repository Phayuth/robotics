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



theta = np.deg2rad(40)
p1 = np.array([[1],[1],[1],[1]]) # the fourth element MUST be equal to 1

H01 = hom_pure_translation(5,0,0)
H12 = hom_pure_translation(0,5,0)
H23 = hom_pure_translation(0,0,5)

H = H01@H12@H23
p0 = H@p1

print(H)
print(p1)
print(p0)