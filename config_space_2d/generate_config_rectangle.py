import os
import sys
wd = os.path.abspath(os.getcwd())
sys.path.append(str(wd))

import numpy as np
from collision_check_geometry.collision_class import sqr_rec_2d_obj, intersect_rectangle_v_rectangle

# This is to demonstate a sampling of state-space of robot

def configuration_generate_rectangle(obs_list):

    sample = np.arange(0,100,1)

    grid = []
    a_row_of_grid = []
    for i in sample:
        for j in sample:

            rec_moving = sqr_rec_2d_obj(i,j,h=2,w=2)

            col = []
            for k in obs_list:
                collision = intersect_rectangle_v_rectangle(rec_moving, k)
                col.append(collision)

            if True in col:
                a_row_of_grid.append(True)
            else:
                a_row_of_grid.append(False)
        grid.append(a_row_of_grid)
        a_row_of_grid = []

    grid_np = np.array(grid).astype(int)

    return 1 - grid_np

if __name__=="__main__":
    import matplotlib.pyplot as plt
    from map.taskmap_geo_format import task_rectangle_obs_2

    obs_list = task_rectangle_obs_2()
    grid_np = configuration_generate_rectangle(obs_list)
    print(grid_np.shape)
    plt.imshow(grid_np)
    plt.grid(True)
    plt.show()