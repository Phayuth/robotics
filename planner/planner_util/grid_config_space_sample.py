import os
import sys
wd = os.path.abspath(os.getcwd())
sys.path.append(str(wd))

import numpy as np
import matplotlib.pyplot as plt
from collision_check_geometry.collision_class import sqr_rec_2d_obj, intersect_rectangle_v_rectangle

# This is to demonstate a sampling of state-space of robot

# Create a sample of possible state-space
sample = np.arange(0,100,1)

h = 3
w = 3

rec1 = sqr_rec_2d_obj(50,50,h,w)

grid = []
a_row_of_grid = []
for i in sample:
    for j in sample:
        
        rec2 = sqr_rec_2d_obj(i,j,h,w)
        collision = intersect_rectangle_v_rectangle(rec1,rec2)
        a_row_of_grid.append(collision)
    grid.append(a_row_of_grid)
    a_row_of_grid = []

grid_np = np.array(grid).astype(int)
print(grid_np.shape)
plt.imshow(grid_np)
plt.grid(True)
plt.show()