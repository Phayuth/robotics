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

if __name__ == "__main__":
    # 2D
    theta = np.deg2rad(90)
    p1 = np.array([[1],[0]])
    p2 = rotation_2d(theta) @ p1
    print("This is 2d rotation")
    print(p1)
    print(p2)

    # 3D
    p1 = np.array([[1],[0],[0]])
    p2 = rotation_3d_y_axis(theta) @ p1
    print("This is 3d rotation")
    print(p1)
    print(p2)

    # Sequence of rotation
    # for concurrent frame rotation, we post multiply of rotation matrix
    theta = np.deg2rad(90)
    p1 = np.array([[1],[0],[0]])
    p2 = rotation_3d_y_axis(theta) @ rotation_3d_z_axis(theta) @ p1
    print("This concurrent frame rotation")
    print(p1)
    print(p2)

    # for fixed frame rotation, we pre multiply of rotation matrix
    theta = np.deg2rad(90)
    p1 = np.array([[1],[0],[0]])
    p2 = rotation_3d_z_axis(theta) @ rotation_3d_y_axis(theta) @ p1
    print("This is fixed frame rotation")
    print(p1)
    print(p2)
