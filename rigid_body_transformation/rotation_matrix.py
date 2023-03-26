import numpy as np

def rot2d(theta):
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta),  np.cos(theta)]])
    return R

def rotx(theta):
    R = np.array([[1,             0,              0],
                  [0, np.cos(theta), -np.sin(theta)],
                  [0, np.sin(theta),  np.cos(theta)]])
    return R

def roty(theta):
    R = np.array([[np.cos(theta),  0,  np.sin(theta)],
                  [            0,  1,              0],
                  [-np.sin(theta), 0,  np.cos(theta)]])
    return R

def rotz(theta):
    R = np.array([[np.cos(theta), -np.sin(theta),  0],
                  [np.sin(theta),  np.cos(theta),  0],
                  [            0,              0,  1]])
    return R

def post_multiply(*argv):
    """
    sequence of rotation
    for concurrent frame rotation, we post multiply of rotation matrix.
        theta = np.deg2rad(90)
        p1 = np.array([[1],[0],[0]])
        p2 = rotation_3d_y_axis(theta) @ rotation_3d_z_axis(theta) @ p1

    """
    rotation_list = []
    for arg in argv:
        rotation_list.append(arg)

    rot = rotation_list[0]
    for i in range(1, len(rotation_list)):
        rot = rot @ rotation_list[i]

    return rot

def pre_multiply(*argv):
    """
    sequence of rotation
    fixed frame rotation, we pre multiply of rotation matrix.
        theta = np.deg2rad(90)
        p1 = np.array([[1],[0],[0]])
        p2 = rotation_3d_z_axis(theta) @ rotation_3d_y_axis(theta) @ p1

    """
    rotation_list = []
    for arg in argv:
        rotation_list.append(arg)

    rot = rotation_list[-1]
    for i in range(-2, -len(rotation_list), -1):
        rot = rotation_list[i] @ rot

    return rot

def rotfixang(seq, angle_seq):
    """rotation in fixed coordinate system. front multiply.
    
    Args:
        seq (str): sequence of rotation. ex: xyz, zyx, xyx, ...
        angle_seq (np.ndarray): the input of angle_seq sequence must be (gamma(x), beta(y), alpha(z))
    """
    if len(seq) != 3:
        return None
    else:
        rotation = np.identity(3)
        for index, value in enumerate(seq):
            if value == 'x':
                rotation = rotx(angle_seq[index,0]) @ rotation
            elif value == 'y':
                rotation = roty(angle_seq[index,0]) @ rotation
            elif value == 'z':
                rotation = rotz(angle_seq[index,0]) @ rotation

        return rotation

def roteuler(seq, angle_seq):
    """rotation in euler coordinate system. back multiply.
    not correct yet
    Args:
        seq (str): sequence of rotation. ex: xyz, zyx, xyx, ...
        angle_seq (np.ndarray): the input of angle_seq sequence must be (gamma(x), beta(y), alpha(z))
    """
    if len(seq) != 3:
        return None
    else:
        rotation = np.identity(3)
        for index, value in enumerate(seq): # reversed oder
            if value == 'x':
                rotation = rotation @ rotx(angle_seq[-index-1,0])
            elif value == 'y':
                rotation = rotation @ roty(angle_seq[-index-1,0])
            elif value == 'z':
                rotation = rotation @ rotz(angle_seq[-index-1,0])

        return rotation

def fixang_to_rotvec(rotmat):
    r11 = rotmat[0,0]
    r12 = rotmat[0,1]
    r21 = rotmat[1,0]
    r22 = rotmat[1,1]
    r31 = rotmat[2,0]
    r32 = rotmat[2,1]
    r33 = rotmat[2,2]

    beta = np.arctan2(-r31,np.sqrt(r11**2 + r21**2))

    if abs(beta - np.pi/2) > 1e-6 and abs(beta + np.pi/2) > 1e-6:
        beta = np.arctan2(-r31,np.sqrt(r11**2 + r21**2))
        alpha = np.arctan2(r21/np.cos(beta), r11/np.cos(beta))
        gamma = np.arctan2(r32/np.cos(beta), r33/np.cos(beta))

    elif beta == np.pi/2: # gimal lock, by convention we choose alpha = 0
        beta = np.pi/2
        alpha = 0
        gamma = np.arctan2(r12, r22)

    elif beta == -np.pi/2: # gimbal lock, by convention we choose alpha = 0
        beta = -np.pi/2
        alpha = 0
        gamma = -np.arctan2(r12, r22)

    return np.array([gamma, beta, alpha]).reshape(3,1)

def euler_to_rotvec(rotmat):
    r11 = rotmat[0,0]
    r12 = rotmat[0,1]
    r13 = rotmat[0,2]
    r23 = rotmat[1,2]
    r31 = rotmat[2,0]
    r32 = rotmat[2,1]
    r33 = rotmat[2,2]

    beta = np.arctan2(np.sqrt(r31**2 + r32**2), r33)

    if 0 < beta < np.pi:
        beta = np.arctan2(np.sqrt(r31**2 + r32**2), r33)
        alpha = np.arctan2(r23/np.sin(beta), r13/np.sin(beta))
        gamma = np.arctan2(r32/np.sin(beta), -r31/np.sin(beta))

    elif beta == 0: # gimbal lock, by convention we choose alpha = 0
        beta = 0
        alpha = 0
        gamma = np.arctan2(-r12, r11)

    elif beta == np.pi: # gimbal lock, by convention we choose alpha = 0
        beta = np.pi
        alpha = 0
        gamma = np.arctan2(r12, -r11)

    return np.array([gamma, beta, alpha]).reshape(3,1)

def axang_to_quat(theta, n):
    # quaternion = q0 + iq1 + jq2 +kq3
    # is a rotation by theta about unit vector n = (nx,ny,nz)
    norm_n = np.linalg.norm(n) # calculate norm of vector
    nx = n[0,0]/norm_n # unit vector in x direction
    ny = n[1,0]/norm_n # unit vector in y direction
    nz = n[2,0]/norm_n # unit vector in z direction

    q0 = np.cos(theta/2)
    q1 = nx*np.sin(theta/2)
    q2 = ny*np.sin(theta/2)
    q3 = nz*np.sin(theta/2)

    return np.array([q0,q1,q2,q3]).reshape(4,1)

def quat_to_axang(q):
    q0 = q[0,0]
    q1 = q[1,0]
    q2 = q[2,0]
    q3 = q[3,0]

    theta = np.arccos(q0)*2
    nx = q1/np.sin(theta/2)
    ny = q2/np.sin(theta/2)
    nz = q3/np.sin(theta/2)

    return theta, np.array([nx,ny,nz]).reshape(3,1)

def rotmat_to_axang(rotmat):
    r11 = rotmat[0,0]
    r12 = rotmat[0,1]
    r13 = rotmat[0,2]
    r21 = rotmat[1,0]
    r22 = rotmat[1,1]
    r23 = rotmat[1,2]
    r31 = rotmat[2,0]
    r32 = rotmat[2,1]
    r33 = rotmat[2,2]

    theta = np.arccos((r11 + r22 + r33 - 1)/2)
    k = (1/(2*np.sin(theta))) * np.array([r32 - r23, r13 - r31, r21 - r12]).reshape(3,1)

    return theta, k

def axang_to_rotmat(theta, k):
    k = k/np.linalg.norm(k)
    kx = k[0,0]
    ky = k[1,0]
    kz = k[2,0]
    vt = 1 - np.cos(theta)

    rotmat = np.array([[   kx*kx*vt + np.cos(theta),  kx*ky*vt - kz*np.sin(theta),  kx*kz*vt + ky*np.sin(theta)],
                       [kx*ky*vt + kz*np.sin(theta),     ky*ky*vt + np.cos(theta),  ky*kz*vt - kx*np.sin(theta)],
                       [kx*kz*vt - ky*np.sin(theta),  ky*kz*vt + kx*np.sin(theta),     kz*kz*vt + np.cos(theta)]])

    return rotmat

def quat_to_rotmat(q):
    q = q / np.linalg.norm(q)
    q0 = q[0,0]
    q1 = q[1,0]
    q2 = q[2,0]
    q3 = q[3,0]

    rotmat = np.array([[1 - 2*q2**2 - 2*q3**2,       2*q1*q2 - 2*q0*q3,      2*q1*q3 + 2*q0*q2],
                       [    2*q1*q2 + 2*q0*q3,   1 - 2*q1**2 - 2*q3**2,      2*q2*q3 - 2*q0*q1],
                       [    2*q1*q3 - 2*q0*q2,       2*q2*q3 + 2*q0*q1,  1 - 2*q1**2 - 2*q2**2]])
    
    return rotmat

def rotmat_to_quat(rotmat):
    r11, r12, r13 = rotmat[0, 0], rotmat[0, 1], rotmat[0, 2]
    r21, r22, r23 = rotmat[1, 0], rotmat[1, 1], rotmat[1, 2]
    r31, r32, r33 = rotmat[2, 0], rotmat[2, 1], rotmat[2, 2]

    q0 = np.sqrt(1 + r11 + r22 + r33) / 2
    q1 = (r32 - r23) / (4 * q0)
    q2 = (r13 - r31) / (4 * q0)
    q3 = (r21 - r12) / (4 * q0)

    return np.array([q0, q1, q2, q3]).reshape(4,1)

def vec_to_skew(x):
    return np.array([[      0,  -x[2,0],  x[1,0]],
                     [ x[2,0],        0, -x[0,0]],
                     [-x[1,0],   x[0,0],       0]])



if __name__ == "__main__":
    np.set_printoptions(suppress=True)

    # fixed axis rotation
    gamma = np.random.uniform(-np.pi, np.pi)
    beta = np.random.uniform(-np.pi, np.pi)
    alpha = np.random.uniform(-np.pi, np.pi)
    gamma_beta_alpha = np.array([gamma,beta,alpha]).reshape(3,1)

    fixed_angle = rotfixang('xyz', gamma_beta_alpha)
    ang = fixang_to_rotvec(fixed_angle)
    fixed_angle_again = rotfixang('xyz', ang)
    print("==>> fixed_angle original: \n", fixed_angle)
    print("==>> fixed_angle inverse problem: \n", fixed_angle_again)


    # equivalent axis angle rotation
    theta = np.random.uniform(-np.pi, np.pi)
    k1 = np.random.uniform(0,1)
    k2 = np.random.uniform(0,1)
    k3 = np.random.uniform(0,1)
    k = np.array([k1,k2,k2]).reshape(3,1)
    
    axangRot = axang_to_rotmat(theta, k)
    thet, kk = rotmat_to_axang(axangRot)
    print("==>> theta original: \n", theta)
    print("==>> k original: \n", k)
    print("==>> theta inverse problem: \n", thet)
    print("==>> k inverse problem: \n", kk)


    # quaternion
    q0 = np.random.uniform(0,1)
    q1 = np.random.uniform(0,1)
    q2 = np.random.uniform(0,1)
    q3 = np.random.uniform(0,1)
    quatt = np.array([q0,q1,q2,q3]).reshape(4,1)
    quatt = quatt/ np.linalg.norm(quatt)
    
    rotfromqt = quat_to_rotmat(quatt)
    qttt = rotmat_to_quat(rotfromqt)
    print("==>> quaternion original: \n", quatt)
    print("==>> quaternion inverse problem: \n", qttt)

    q = axang_to_quat(theta, k)
    theta_ori, n_ori = quat_to_axang(q)
    print("==>> theta_ori: \n", theta_ori)
    print("==>> n_ori: \n", n_ori)


    # skew matrix
    x = np.array([[1],
                  [0],
                  [0]])

    print(vec_to_skew(x))