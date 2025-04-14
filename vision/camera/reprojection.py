import cv2
import numpy as np

fx = 598.460339
fy = 597.424060
cx = 317.880979
cy = 233.262422

# distorstion k1, k2, p1, p2, k3 params order
k1 = 0.142729
k2 = -0.282139

p1 = -0.005699
p2 = -0.012027

k3 = 0.000000

# Define the intrinsic camera matrix
camera_matrix = np.array([[fx,  0, cx],
                          [ 0, fy, cy],
                          [ 0,  0,  1]], dtype=np.float32)

# Distortion coefficients (k1, k2, p1, p2, k3)
dist_coeffs = np.array([k1, k2, p1, p2, k3], dtype=np.float32)

x = 0.38566032
y = 0.11239833
z = 0.0
undistorted_point = np.array([[[x, y, z]]], dtype=np.float32)

# Distort the point using OpenCV's projectPoints function
distorted_point = cv2.projectPoints(undistorted_point, np.zeros((3, 1)), np.zeros((3, 1)), camera_matrix, dist_coeffs)[0]
print(f"> distorted_point: {distorted_point}")
