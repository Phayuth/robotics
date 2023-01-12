import numpy as np
import matplotlib.pyplot as plt

theta1 = np.pi/2
theta2 = 0
theta3 = 0
a1 = 2
a2 = 2
a3 = 2


H01 = np.array([[np.cos(theta1),    -np.sin(theta1),    0,    0],
                [np.sin(theta1),     np.cos(theta1),    0,    0],
                [             0,                  0,    1,    0],
                [             0,                  0,    0,    1]])

H12 = np.array([[np.cos(theta2),    -np.sin(theta2),    0,    a1],
                [np.sin(theta2),     np.cos(theta2),    0,     0],
                [             0,                  0,    1,     0],
                [             0,                  0,    0,     1]])

H23 = np.array([[np.cos(theta3),    -np.sin(theta3),    0,    a2],
                [np.sin(theta3),     np.cos(theta3),    0,     0],
                [             0,                  0,    1,     0],
                [             0,                  0,    0,     1]])
    
H34 = np.array([[             1,                  0,    0,    a3],
                [             0,                  1,    0,     0],
                [             0,                  0,    1,     0],
                [             0,                  0,    0,     1]])

H04 = H01 @ H12 @ H23 @ H34

p4 = np.array([[0],[0],[0],[1]])

p0 = H04 @ p4

print(p0)