import numpy as np

def collinearity_equation_with_principal_point(X, Y, Z, X0, Y0, Z0, R, f, x0, y0):
    # Relative coordinates with respect to the camera center
    dX = X - X0
    dY = Y - Y0
    dZ = Z - Z0

    # Apply rotation matrix to the 3D point
    rotated_point = np.dot(R, np.array([dX, dY, dZ]))

    # Calculate image coordinates with the principal point correction
    # x = x0 - f * (rotated_point[0] / rotated_point[2])
    # y = y0 - f * (rotated_point[1] / rotated_point[2])
    x = x0 + f * (rotated_point[0] / rotated_point[2])
    y = y0 + f * (rotated_point[1] / rotated_point[2])

    return x, y


def camera_projection_matrix_with_principal_point(X, Y, Z, K, R, t):
    # Convert the 3D point into homogeneous coordinates
    world_point = np.array([X, Y, Z, 1])

    # Projection matrix (K[R|t])
    extrinsic_matrix = np.hstack((R, t.reshape(-1, 1)))  # [R | t]
    P = np.dot(K, extrinsic_matrix)  # Full projection matrix

    # Project the point into 2D homogeneous coordinates
    image_point_h = np.dot(P, world_point)

    # Normalize to get the image coordinates (in pixels)
    x = image_point_h[0] / image_point_h[2]
    y = image_point_h[1] / image_point_h[2]

    return x, y


# Principal point coordinates
x0, y0 = 320, 240  # Example: center of a 640x480 image

# Example rotation matrix (identity for simplicity) and parameters
R = np.eye(3)  # No rotation (for simplicity)
f = 1000       # Focal length
X0, Y0, Z0 = 0, 0, 0  # Camera position
X, Y, Z = 10, 20, 50  # 3D point

# Call the collinearity equation function with the principal point
x_collinearity_pp, y_collinearity_pp = collinearity_equation_with_principal_point(X, Y, Z, X0, Y0, Z0, R, f, x0, y0)
print(f"Collinearity equation with principal point: x = {x_collinearity_pp}, y = {y_collinearity_pp}")

# Intrinsic matrix with principal point
K = np.array([[f, 0, x0],  # Focal length in x-direction and principal point x0
              [0, f, y0],  # Focal length in y-direction and principal point y0
              [0, 0, 1]])

t = np.array([X0, Y0, Z0])  # No translation (for simplicity)

x_projection_pp, y_projection_pp = camera_projection_matrix_with_principal_point(X, Y, Z, K, R, t)
print(f"Projection matrix with principal point: x = {x_projection_pp}, y = {y_projection_pp}")
