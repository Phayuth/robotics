import os
import sys
wd = os.path.abspath(os.getcwd())
sys.path.append(str(wd))

import numpy as np
import matplotlib.pyplot as plt
from collision_check_geometry.collision_class import obj_rec, obj_line2d
from taskmap_img_format import map_2d_1, map_2d_2, pmap, bmap



map = map_2d_1()
map = pmap()
plt.imshow(map)
plt.show()

size_x = map.shape[0]
size_y = map.shape[1]
cell_size = 2*np.pi/size_x

obj = [obj_rec(i*cell_size,j*cell_size,cell_size,cell_size) for i in range(size_x) for j in range(size_y) if map[i,j] == 1]
for o in obj:
    o.plot()
plt.show()
