import numpy as np

def rotation_2d(theta):
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta),  np.cos(theta)]])
    return R

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

def concurrent_frame_rotation(*argv):
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

def fixed_frame_rotation(*argv):
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

def fixed_angle_rotation(seq, angle_seq):
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
                rotation = rotation_3d_x_axis(angle_seq[index,0]) @ rotation
            elif value == 'y':
                rotation = rotation_3d_y_axis(angle_seq[index,0]) @ rotation
            elif value == 'z':
                rotation = rotation_3d_z_axis(angle_seq[index,0]) @ rotation

        return rotation

def fixed_angle_rotation_to_rotation_vec(rotation_matrix):
    pass

def euler_angle_rotation(seq, angle_seq):
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
        for index, value in reversed(list(enumerate(seq))): # reversed oder
            print(index , value)
            if value == 'x':
                rotation = rotation_3d_x_axis(angle_seq[index,0]) @ rotation
            elif value == 'y':
                rotation = rotation_3d_y_axis(angle_seq[index,0]) @ rotation
            elif value == 'z':
                rotation = rotation_3d_z_axis(angle_seq[index,0]) @ rotation

        return rotation

def euler_angle_rotation_to_rotation_vec(rotation_matrix):
    pass



if __name__ == "__main__":
    np.set_printoptions(suppress=True)

    # # 2D
    # theta = np.deg2rad(90)
    # p1 = np.array([[1],[0]])
    # p2 = rotation_2d(theta) @ p1
    # print("This is 2d rotation")
    # print(p1)
    # print(p2)

    # # 3D
    # p1 = np.array([[1],[0],[0]])
    # p2 = rotation_3d_y_axis(theta) @ p1
    # print("This is 3d rotation")
    # print(p1)
    # print(p2)

    # sequence of rotation
    # for concurrent frame rotation, we post multiply of rotation matrix
    # theta = np.deg2rad(90)
    # confrot = concurrent_frame_rotation(rotation_3d_x_axis(theta), rotation_3d_y_axis(theta), rotation_3d_z_axis(theta))
    # print(confrot)
    
    # for fixed frame rotation, we pre multiply of rotation matrix
    # fixfrot = fixed_frame_rotation(rotation_3d_z_axis(theta), rotation_3d_y_axis(theta), rotation_3d_x_axis(theta))
    # print(fixfrot)


    # fixed angle rotation
    gamma = np.pi/2
    beta = np.pi/2
    alpha = np.pi/4
    gamma_beta_alpha = np.array([gamma,beta,alpha]).reshape(3,1)

    # fixed_angle_rot = fixed_angle_rotation('xyz', gamma_beta_alpha)
    # print("==>> fixed_angle_rot: \n", fixed_angle_rot)

    # fixed_angle_rot_check = rotation_3d_z_axis(alpha) @ rotation_3d_y_axis(beta) @ rotation_3d_x_axis(gamma)
    # print("==>> fixed_angle_rot_check: \n", fixed_angle_rot_check)

    euler_angle_rot = euler_angle_rotation('zyx', gamma_beta_alpha)
    print("==>> euler_angle_rot: \n", euler_angle_rot)