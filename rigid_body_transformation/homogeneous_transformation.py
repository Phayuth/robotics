import numpy as np
np.set_printoptions(suppress=True)

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
    check = gs @ gs_iv # return Identity Matrix

    tac = np.array([[ np.cos(np.deg2rad(90)),np.cos(np.deg2rad(120)), np.cos(np.deg2rad(30)), 3],
                    [ np.cos(np.deg2rad(90)), np.cos(np.deg2rad(30)), np.cos(np.deg2rad(60)), 0],
                    [np.cos(np.deg2rad(180)), np.cos(np.deg2rad(90)), np.cos(np.deg2rad(90)), 2],
                    [                      0,                      0,                      0, 1]])
    tca = inverse_hom_trans(tac)
    tac_rotation_mat = tac[0:3,0:3]
    tca_rotation_mat = tca[0:3,0:3]

    tcb = np.array([[ np.cos(np.deg2rad(90+90-36.9)),np.cos(np.deg2rad(90+36.9)), np.cos(np.deg2rad(90)), 3],
                    [ np.cos(np.deg2rad(90)), np.cos(np.deg2rad(90)), np.cos(np.deg2rad(0)), 2],
                    [np.cos(np.deg2rad(90+36.9)), np.cos(np.deg2rad(36.9)), np.cos(np.deg2rad(90)), 0],
                    [                      0,                      0,                      0, 1]])
    tcb_rotation_mat = tcb[0:3,0:3]
