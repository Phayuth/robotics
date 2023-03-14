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
                  [0,              0,              0,    1]])
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
    
def inverse_hom_trans(hom_trans):
    R = np.array([[hom_trans[0,0], hom_trans[0,1], hom_trans[0,2]],
                  [hom_trans[1,0], hom_trans[1,1], hom_trans[1,2]],
                  [hom_trans[2,0], hom_trans[2,1], hom_trans[2,2]]])
    
    P = np.array([[hom_trans[0,3]],
                  [hom_trans[1,3]],
                  [hom_trans[2,3]]])

    upper = np.hstack((R.T,-R.T @ P))
    lower = np.array([[0,0,0,1]])

    T_inv = np.vstack((upper,lower))

    return T_inv

if __name__ == "__main__":
    gs = hom_rotation_x_axis(theta=1)

    gs_iv = inverse_hom_trans(gs)
    print("==>> gs_iv: \n", gs_iv)

    check = gs @ gs_iv # return Identity Matrix
    print("==>> check: \n", check)