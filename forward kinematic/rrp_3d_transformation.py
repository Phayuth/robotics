import numpy as np
import matplotlib.pyplot as plt

theta1 = 0
theta2 = 0
a1 = 2
a2 = 2
a3 = 2
a4 = 2
d3 = 1

H_1_to_0 = np.array([[np.cos(theta1),                  0,    np.sin(theta1),    a2*np.cos(theta1)],
                     [np.sin(theta1),                  0,   -np.cos(theta1),    a2*np.sin(theta1)],
                     [             0,                  1,                 0,                   a1],
                     [             0,                  0,                 0,                   1]])

H_2_to_1 = np.array([[             0,    -np.sin(theta2),    np.cos(theta2),    a3*np.cos(theta2)],
                     [             0,     np.cos(theta2),    np.sin(theta2),    a3*np.sin(theta2)],
                     [            -1,                  0,                 0,                    0],
                     [             0,                  0,                 0,                    1]])

H_3_to_2 = np.array([[             1,                  0,                 0,                    0],
                     [             0,                  1,                 0,                    0],
                     [             0,                  0,                 1,                a4+d3],
                     [             0,                  0,                 0,                    1]])            

H_3_to_0 = H_1_to_0 @ H_2_to_1 @ H_3_to_2

p3 = np.array([[0],[0],[0],[1]])

p0 = H_3_to_0 @ p3

print(p0)