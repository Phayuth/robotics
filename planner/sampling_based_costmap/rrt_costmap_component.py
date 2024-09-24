import os
import sys

sys.path.append(str(os.path.abspath(os.getcwd())))

import time
import numpy as np
from spatial_geometry.map_class import map_val


class Node:

    def __init__(self, config, parent=None, cost=0.0) -> None:
        self.config = config
        self.parent = parent
        self.child = []
        self.cost = cost

    def __repr__(self) -> str:
        return f'\nconfig = {self.config.T}, hasParent = {True if self.parent != None else False}, NumChild = {len(self.child)}'


class RRTComponent:

    def __init__(self, baseConfigFile) -> None:
        pass

    def bias_sampling(map):

        row = map.shape[0]
        p = np.ravel(map) / np.sum(map)
        x_sample = np.random.choice(len(p), p=p)
        x = x_sample // row
        y = x_sample % row
        x = np.random.uniform(low=x - 0.5, high=x + 0.5)
        y = np.random.uniform(low=y - 0.5, high=y + 0.5)
        return x, y

    def Sampling(self):
        """bias sampling method

        Returns:
            node: random bias sampling node
        """
        height = self.map.shape[0]
        width = self.map.shape[1]
        depth = self.map.shape[2]

        p = np.ravel(self.map) / np.sum(self.map)

        x_sample = np.random.choice(len(p), p=p)

        z = x_sample // (width * height)
        x = (x_sample - z*(width * height)) % width
        y = (x_sample - z*(width * height)) // width

        x = np.random.uniform(low=x - 0.5, high=x + 0.5)
        y = np.random.uniform(low=y - 0.5, high=y + 0.5)
        z = np.random.uniform(low=z - 0.5, high=z + 0.5)

        x_rand = node(x, y, z)

        return x_rand

    def sample_point_on_hypersphere(center, radius, dimension):
        random_numbers = np.random.normal(size=dimension)
        norm = np.linalg.norm(random_numbers)
        point = random_numbers / norm
        scaled_point = point * radius
        moved_point = scaled_point + center
        return moved_point

    def distance(q_base, q_far):
        return np.linalg.norm([q_far.x - q_base.x, q_far.y - q_base.y])

    def segmented_line(q_base, q_far, num_seg=10):
        x1 = q_base.x
        y1 = q_base.y
        x2 = q_far.x
        y2 = q_far.y

        distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        segment_length = distance / (num_seg-1)
        points = [q_base]
        for i in range(1, num_seg - 1):
            x = x1 + i * segment_length * (q_far.x - q_base.x) / distance
            y = y1 + i * segment_length * (q_far.y - q_base.y) / distance
            point = node(x, y)
            points.append(point)
        points.append(q_far)

        return points

    def getcost(map, q, minmax_q=[[-np.pi, np.pi], [-np.pi, np.pi]], return_index_cost=False):

        if minmax_q[0][0] < q.x < minmax_q[0][1] or minmax_q[1][0] < q.y < minmax_q[1][1]:  # value must inside the range, not even equal pi

            ind_x = int((map_val(q.x, minmax_q[0][0], minmax_q[0][1], 0, map.shape[0])))
            ind_y = int((map_val(q.y, minmax_q[1][0], minmax_q[1][1], 0, map.shape[1])))

            if return_index_cost:
                return ind_x, ind_y, map[ind_y, ind_x]
            else:
                return map[ind_y, ind_x]  # in img format row is y, column is x, spend weeks cause of this

    def setcost(map, q, cost, minmax_q=[[-np.pi, np.pi], [-np.pi, np.pi]]):
        cell_size_x = (minmax_q[0][1] - minmax_q[0][0]) / map.shape[0]
        cell_size_y = (minmax_q[1][1] - minmax_q[1][0]) / map.shape[1]

        ind_x = int((q.x / cell_size_x))
        ind_y = int((q.y / cell_size_y))

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
        map_overlay = np.ones((30, 30))
        seg_length = 1
        seg_point = int(np.ceil(distance(start, end) / seg_length))

        costv = []
        if seg_point > 1:
            v = np.array([end.x - start.x, end.y - start.y]) / (seg_point)

            for i in range(seg_point + 1):
                seg = np.array([start.x, start.y]) + i*v
                seg = np.around(seg)
                map_overlay[int(seg[1]), int(seg[0])] = 0
                if cost := 1 - costmap[int(seg[1]), int(seg[0])] == 1:
                    cost = 1e10
                costv.append(cost)
            cost = sum(costv) / (seg_point+1)
            return map_overlay, cost

        else:
            map_overlay[int(start.y), int(start.x)] = 0
            map_overlay[int(end.y), int(end.x)] = 0
            cost = ((costmap[int(start.y), int(start.x)] + costmap[int(end.y), int(end.x)])) / 2

            return map_overlay, cost

    def Line_Cost(self, start:node, end:node)-> float:
        """calculate line cost. if the path crosses an obstacle, it has a high cost

        Args:
            start (node): start pose with node class
            end (node): end pose with node class

        Returns:
            float: line cost = (distance weight * cost distance) + (obstacle weight * obstacle distance)
        """

        cost = self.w1*(self.Distance_Cost(start, end)/(self.eta)) + self.w2*(self.Obstacle_Cost(start, end))

        return cost


    def Obstacle_Cost(self, start:node, end:node)-> float:
        """calculate cost for obstacle node

        Args:
            start (node): start pose with node class
            end (node): end pose with node class

        Returns:
            float: cost of obstacle
        """
        seg_length = 1
        seg_point = int(np.ceil(np.linalg.norm(start.arr - end.arr) / seg_length))

        value = 0
        if seg_point > 1:
            v = (end.arr - start.arr) / (seg_point)

            for i in range(seg_point+1):
                seg = start.arr + i * v
                seg = np.around(seg)
                if 1 - self.map[int(seg[0]), int(seg[1]), int(seg[2])] == 1:
                    cost = 1e10  # edge pass through the hard obstacle

                    return cost
                else:
                    value += 1 - self.map[int(seg[0]), int(seg[1]), int(seg[2])]

            cost = value / (seg_point+1)

            return cost

        else:

            value = self.map[int(start.arr[0]), int(start.arr[1]), int(start.arr[2])] + self.map[int(end.arr[0]), int(end.arr[1]), int(end.arr[2])]
            cost = value / 2

            return cost