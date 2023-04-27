""" This is to demonstate a sampling of state-space of robot
- Robot Type : Rectangle
- DOF : 2
- Taskspace map : Geometry Format
- Collision Check : Geometry Based
"""

import os
import sys

wd = os.path.abspath(os.getcwd())
sys.path.append(str(wd))

import numpy as np
from collision_check_geometry.collision_class import ObjRec, intersect_rectangle_v_rectangle


def configuration_generate_rectangle(obsList):
    sample = np.arange(0, 100, 1)
    grid = []
    aRowOfGrid = []
    for i in sample:
        for j in sample:
            recMoving = ObjRec(i, j, h=2, w=2)
            col = []
            for k in obsList:
                collision = intersect_rectangle_v_rectangle(recMoving, k)
                col.append(collision)
            if True in col:
                aRowOfGrid.append(True)
            else:
                aRowOfGrid.append(False)
        grid.append(aRowOfGrid)
        aRowOfGrid = []
    gridNp = np.array(grid).astype(int)
    return 1 - gridNp


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from map.taskmap_geo_format import task_rectangle_obs_2

    # SECTION - configuration space
    obsList = task_rectangle_obs_2()
    gridNp = configuration_generate_rectangle(obsList)
    plt.imshow(gridNp)
    plt.grid(True)
    plt.show()