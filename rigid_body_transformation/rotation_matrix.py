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
    theta = np.deg2rad(90)
    confrot = concurrent_frame_rotation(rotation_3d_x_axis(theta), rotation_3d_y_axis(theta), rotation_3d_z_axis(theta))
    print(confrot)
    
    # for fixed frame rotation, we pre multiply of rotation matrix
    fixfrot = fixed_frame_rotation(rotation_3d_z_axis(theta), rotation_3d_y_axis(theta), rotation_3d_x_axis(theta))
    print(fixfrot)