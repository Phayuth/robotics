import os
import sys
wd = os.path.abspath(os.getcwd())
sys.path.append(str(wd))

from robot.ur5e import ur5e
import numpy as np

# create robot class
robot = ur5e()

# transformation from camera frame to ee frame
T6e = np.array([[-1, 0, 0, 3.3],
                [0, -1, 0,  10],
                [0,  0, 1,  -3],
                [0,  0, 0,   1]])

# transformation from camera frame to base frame
theta = (np.pi/180) * np.array([-233, -6, -37, -137, 98, -180]).reshape(6,1)
T06 = robot.forward_kinematic(theta, return_full_H=True)
T0e = T06 @ T6e

# read point cloud from file (the data is measure relative to camera frame)
xyz_cam_frame = np.load()

for index, value in enumerate(xyz_cam_frame):
    xyz_base_frame = T0e @ value