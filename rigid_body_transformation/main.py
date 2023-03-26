import numpy as np
from homogeneous_transformation import *
from rotation_matrix import *
from plot_frame import *

# homog transform 
theta = np.pi/4

p1 = np.array([[1],[1],[1],[1]]) # the fourth element MUST be equal to 1

H01 = htrn(5,0,0)
H12 = htrn(0,5,0)
H23 = hrotz(theta)
H = H01 @ H12 @ H23
p0 = H @ p1

# plot frame 
# 2D
rot = rot2d(theta)
tran = np.array([[1],[1]])
plot_frame_2d(rot,tran, plt_basis=True, plt_show=True)

# 3D
gs = hrotz(theta)
plot_frame_3d(gs, plt_basis=True, plt_show=True)
