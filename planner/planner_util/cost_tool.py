import os
import sys
wd = os.path.abspath(os.getcwd())
sys.path.append(str(wd))

import numpy as np
from node_tool import node
from map.mapclass import map_val


def distance(q_base, q_far):
    return np.linalg.norm([q_far.x - q_base.x, q_far.y - q_base.y])


def segmented_line(q_base, q_far, num_seg=10):
    x1 = q_base.x
    y1 = q_base.y
    x2 = q_far.x
    y2 = q_far.y

    distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    segment_length = distance / (num_seg - 1)
    points = [q_base]
    for i in range(1, num_seg - 1):
        x = x1 + i * segment_length * (q_far.x - q_base.x)/distance
        y = y1 + i * segment_length * (q_far.y - q_base.y)/distance
        point = node(x, y)
        points.append(point)
    points.append(q_far)

    return points


def getcost(map, q, minmax_q = [[-np.pi, np.pi],[-np.pi, np.pi]], return_index_cost=False):

    if minmax_q[0][0] < q.x < minmax_q[0][1] or minmax_q[1][0] < q.y < minmax_q[1][1]: # value must inside the range, not even equal pi
    
        ind_x = int((map_val(q.x, minmax_q[0][0], minmax_q[0][1], 0, map.shape[0])))
        ind_y = int((map_val(q.y, minmax_q[1][0], minmax_q[1][1], 0, map.shape[1])))

        if return_index_cost:
            return ind_x, ind_y, map[ind_y, ind_x]
        else:
            return map[ind_y, ind_x] # in img format row is y, column is x, spend weeks cause of this


def setcost(map, q, cost, minmax_q = [[-np.pi, np.pi],[-np.pi, np.pi]]):
    cell_size_x = (minmax_q[0][1] - minmax_q[0][0])/map.shape[0]
    cell_size_y = (minmax_q[1][1] - minmax_q[1][0])/map.shape[1]

    ind_x = int((q.x/cell_size_x))
    ind_y = int((q.y/cell_size_y))

    map[ind_y, ind_x] = cost


def line_cost(map, q_base, q_far, num_seg, return_index_cost=False):
    seg = segmented_line(q_base, q_far, num_seg=num_seg)
    costlist = [getcost(map, q) for q in seg]
    cost = np.sum(costlist) / num_seg

    if return_index_cost:
        costlist = [getcost(map, q, return_index_cost=True) for q in seg]
        return costlist
    else:
        return cost


def obstacle_cost(costmap, start, end):
    map_overlay = np.ones((30,30))
    seg_length = 1
    seg_point = int(np.ceil(distance(start, end) / seg_length))

    costv = []
    if seg_point > 1:
        v = np.array([end.x - start.x, end.y - start.y]) / (seg_point)

        for i in range(seg_point + 1):
            seg = np.array([start.x, start.y]) + i*v
            seg = np.around(seg)
            map_overlay[int(seg[1]), int(seg[0])] = 0
            if cost:= 1 - costmap[int(seg[1]), int(seg[0])] == 1:
                cost = 1e10
            costv.append(cost)
        cost = sum(costv) / (seg_point+1)
        return map_overlay, cost

    else:
        map_overlay[int(start.y), int(start.x)] = 0
        map_overlay[int(end.y), int(end.x)] = 0
        cost = ((costmap[int(start.y), int(start.x)] + costmap[int(end.y), int(end.x)]))/2

        return map_overlay, cost

if __name__ == "__main__":
    from map.taskmap_img_format import map_2d_empty, map_2d_1
    import matplotlib.pyplot as plt


    # SECTION - obstacle cost
    map = map_2d_1()
    map = np.ones((30,30))
    map[15,15] = 0

    
    q_start = node(5,5)
    q_end = node(6,6)

    mapover, cost = obstacle_cost(map, q_start, q_end)
    print(f"==>> cost: \n{cost}")

    plt.imshow(map)
    plt.imshow(mapover, cmap='gray',alpha=0.5)
    plt.show()
