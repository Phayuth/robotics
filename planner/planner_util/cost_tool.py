import os
import sys
wd = os.path.abspath(os.getcwd())
sys.path.append(str(wd))

import numpy as np
from node_tool import node

def distance(q_base, q_far):
    return np.linalg.norm([q_far.x - q_base.x, q_far.y - q_base.y])

def obstacle_cost(map, start, end):
    seg_length = 1
    seg_point = int(np.ceil(distance(start, end) / seg_length))

    value = 0
    if seg_point > 1:
        v = distance(start, end)/ (seg_point)

        for i in range(seg_point + 1):
            seg = np.array([start.x, start.y]) + i * v
            seg = np.around(seg)
            if 1 - map[int(seg[0]), int(seg[1])] == 1 :
                cost = 1e10
                return cost
            else:
                value += 1 - map[int(seg[0]), int(seg[1])]
        cost = value / (seg_point + 1)
        return cost
    else:
        value = map[int(start.x), int(start.y)] + map[int(end.x), int(end.y)]
        cost = value / 2
        return cost
    
if __name__=="__main__":
    from map.taskmap_img_format import map_2d_empty, map_2d_1
    import matplotlib.pyplot as plt

    map = map_2d_1()
    q_start = node(5,5)
    q_end = node(4,10)
    cost = obstacle_cost(map, q_start, q_end)
    print("==>> cost: \n", cost)

    plt.imshow(map)
    plt.scatter([q_start.x, q_end.x], [q_start.y, q_end.y])
    plt.plot([q_start.x, q_end.x], [q_start.y, q_end.y])
    plt.show()
