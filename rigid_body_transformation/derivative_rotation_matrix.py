import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).parent.parent))

import numpy as np
import rotation_matrix, skew_matrix

if __name__ == "__main__":
    # Vector basis
    x = np.array([[1],
                  [0],
                  [0]])

    y = np.array([[0],
                  [1],
                  [0]])

    z = np.array([[0],
                  [0],
                  [1]])

    # Derivative of rotation matrix about x = skew matrix of x basis vector @ Rotation Matrix x
    theta = 0
    R_dot_x = skew_matrix.skew(x) @ rotation_matrix.rotation_3d_x_axis(theta)
    print(R_dot_x)

    # with omega, omega = (i vector basis)@theta_dot
    theta_dot = 1
    omega = theta_dot * x
    R_dot_x = skew_matrix.skew(omega) @ rotation_matrix.rotation_3d_x_axis(theta)
    print(R_dot_x)