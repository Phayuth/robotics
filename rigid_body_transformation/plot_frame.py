import numpy as np
import matplotlib.pyplot as plt
ax = plt.axes(projection='3d')

def plot_frame_2d(rotmat_2d, translation, plt_basis=False, plt_show=False):

    r1 = np.array([[0],[0]])
    r2 = np.array([[1],[0]])
    r4 = np.array([[0],[1]])

    dx = translation[0,0]
    dy = translation[1,0]

    d1 = np.array([[dx],[dy]])
    d2 = np.array([[dx],[dy]])
    d4 = np.array([[dx],[dy]])

    # rnew = Rotation @ rold + d
    r1new = rotmat_2d @ r1 + d1
    r2new = rotmat_2d @ r2 + d2
    r4new = rotmat_2d @ r4 + d4

    plt.axes().set_aspect('equal')
    if plt_basis:
        # plot basic axis
        plt.axvline(x=0, c="black")
        plt.axhline(y=0, c="black")

    # plot frame
    plt.plot([r1new[0,0], r2new[0,0]],[r1new[1,0], r2new[1,0]],"blue", linewidth=4) #x axis
    plt.plot([r1new[0,0], r4new[0,0]],[r1new[1,0], r4new[1,0]],"red", linewidth=4)  #y axis

    if plt_show:
        plt.show()

def plot_frame_3d(homtran, plt_basis=False, plt_show=False):
    # input 4x4 transform matrix
    rotation = np.array([[homtran[0,0], homtran[0,1], homtran[0,2]],
                         [homtran[1,0], homtran[1,1], homtran[1,2]],
                         [homtran[2,0], homtran[2,1], homtran[2,2]]])
    
    d = np.array([[homtran[0,3]],
                  [homtran[1,3]],
                  [homtran[2,3]]])

    r1 = np.array([[1],[0],[0]])
    r2 = np.array([[0],[1],[0]])
    r3 = np.array([[0],[0],[1]])
    r4 = np.array([[0],[0],[0]])

    r1new = rotation @ r1 + d
    r2new = rotation @ r2 + d
    r3new = rotation @ r3 + d
    r4new = rotation @ r4 + d

    if plt_basis:
        # plot basic axis
        ax.plot3D([0, 2], [0, 0], [0, 0], 'red', linewidth=4)
        ax.plot3D([0, 0], [0, 2], [0, 0], 'purple', linewidth=4)
        ax.plot3D([0, 0], [0, 0], [0, 2], 'green', linewidth=4)

    # plot frame
    ax.plot3D([r4new[0,0], r1new[0,0]], [r4new[1,0], r1new[1,0]], [r4new[2,0], r1new[2,0]], 'gray', linewidth=4, label="gray is x")
    ax.plot3D([r4new[0,0], r2new[0,0]], [r4new[1,0], r2new[1,0]], [r4new[2,0], r2new[2,0]], 'blue', linewidth=4, label="blue is y")
    ax.plot3D([r4new[0,0], r3new[0,0]], [r4new[1,0], r3new[1,0]], [r4new[2,0], r3new[2,0]], 'yellow', linewidth=4, label="yellow is z")
    ax.legend()

    if plt_show:
        plt.show()