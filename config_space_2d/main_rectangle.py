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

rec1 = sqr_rec_2d_obj(50,50,h=3,w=3)
rec2 = sqr_rec_2d_obj(40,20,h=9,w=1)
rec3 = sqr_rec_2d_obj(70,90,h=5,w=2)

grid = []
a_row_of_grid = []
for i in sample:
    for j in sample:

        rec_moving = sqr_rec_2d_obj(i,j,h=2,w=2)

        collision_1 = intersect_rectangle_v_rectangle(rec1,rec_moving)
        collision_2 = intersect_rectangle_v_rectangle(rec2,rec_moving)
        collision_3 = intersect_rectangle_v_rectangle(rec3,rec_moving)

        if collision_1 or collision_2 or collision_3:
            collision = True
        else:
            collision = False

        a_row_of_grid.append(collision)
    grid.append(a_row_of_grid)
    a_row_of_grid = []

grid_np = np.array(grid).astype(int)
print(grid_np.shape)
plt.imshow(grid_np)
plt.grid(True)
plt.show()