import numpy as np
import homogeous_transformation
import rotation_matrix
import matplotlib.pyplot as plt
ax = plt.axes(projection='3d')

def plot_frame_2d(rotation_matrix_2d, translation):

    r1 = np.array([[0],[0]])
    r2 = np.array([[1],[0]])
    r4 = np.array([[0],[1]])

    dx = translation[0,0]
    dy = translation[1,0]

    d1 = np.array([[dx],[dy]])
    d2 = np.array([[dx],[dy]])
    d4 = np.array([[dx],[dy]])

    # rnew = Rotation @ rold + d
    r1new = rotation_matrix_2d @ r1 + d1
    r2new = rotation_matrix_2d @ r2 + d2
    r4new = rotation_matrix_2d @ r4 + d4

    plt.axes().set_aspect('equal')
    plt.axvline(x=0, c="black")
    plt.axhline(y=0, c="black")

    plt.plot([r1new[0,0], r2new[0,0]],[r1new[1,0], r2new[1,0]],"blue", linewidth=4) #x axis
    plt.plot([r1new[0,0], r4new[0,0]],[r1new[1,0], r4new[1,0]],"red", linewidth=4)  #y axis

    plt.show()

def plot_frame_3d(homogeneous_transformation_matrix):
    plt.figure(figsize=(10,10))
    plt.axes().set_aspect('equal')
    plt.axvline(x=0, c="black")
    plt.axhline(y=0, c="black")

    # plot basic axis
    ax.plot3D([0, 0.5], [0, 0], [0, 0], 'red', linewidth=4)
    ax.plot3D([0, 0], [0, 0.5], [0, 0], 'purple', linewidth=4)
    ax.plot3D([0, 0], [0, 0], [0, 0.5], 'green', linewidth=4)

    plt.show()





# rot = rotation_matrix.rotation_2d(-0.5)
# tran = np.array([[1],[1]])

# plot_frame_2d(rot,tran)

plot_frame_3d(0)