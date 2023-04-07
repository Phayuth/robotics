import os
import sys

wd = os.path.abspath(os.getcwd())
sys.path.append(str(wd))

import numpy as np
import matplotlib.pyplot as plt
import time

from map.map_value_range import map_val


class node:
    def __init__(self, x: float, y: float, cost: float = 0, parent = None):
        self.x = x
        self.y = y
        self.cost = cost
        self.parent = parent


class RobotRRTStarCostMapUniSampling:
    def __init__(
        self,
        map: np.ndarray,
        x_init: node,
        x_goal: node,
        w1: float,
        w2: float,
        eta: float = None,
        maxiteration: int = 1000,
    ):
        # map properties
        self.map = map
        self.x_init = node(x_init[0, 0], x_init[1, 0])
        self.x_goal = node(x_goal[0, 0], x_goal[1, 0])
        self.nodes = [self.x_init]

        # planner properties
        self.maxiteration = maxiteration
        # self.m = map.shape[0] * map.shape[1]
        # self.r = (2 * (1 + 1 / 2) ** (1 / 2)) * (self.m / np.pi) ** (1 / 2)
        # self.eta = self.r * (np.log(self.maxiteration) / self.maxiteration) ** (1 / 2)
        self.eta = 0.3
        self.w1 = w1
        self.w2 = w2

        self.sample_taken = 0
        self.total_iter = 0
        self.graph_sample_num = 0
        self.graph_data = np.array([[0, 0]])

        # timing
        self.s = time.time()
        self.e = None
        self.sampling_elapsed = 0
        self.addparent_elapsed = 0
        self.rewire_elapsed = 0

    def uniform_sampling(self) -> node:
        x = np.random.uniform(low=-np.pi, high=np.pi)
        y = np.random.uniform(low=-np.pi, high=np.pi)

        x_rand = node(x, y)

        return x_rand

    def distance_cost(self, start: node, end: node) -> float:
        distance_cost = np.linalg.norm([start.x - end.x, start.y - end.y])

        return distance_cost

    # def obstacle_cost(self, start: float, end: float) -> float:
    #     seg_length = 1
    #     seg_point = int(np.ceil(self.distance_cost(start, end) / seg_length))

    #     value = 0
    #     if seg_point > 1:
    #         v = np.array([end.x - start.x, end.y - start.y]) / (seg_point)

    #         for i in range(seg_point + 1):
    #             seg = np.array([start.x, start.y]) + i * v
    #             seg = np.around(seg)
    #             if 1 - self.map[int(seg[1]), int(seg[0])] == 1:
    #                 cost = 1e10

    #                 return cost
    #             else:
    #                 value += 1 - self.map[int(seg[1]), int(seg[0])]

    #         cost = value / (seg_point + 1)

    #         return cost

    #     else:
    #         value = (self.map[int(start.y), int(start.x)] + self.map[int(end.y), int(end.x)])
    #         cost = value / 2

    #         return cost

    def segmented_line(self, q_base, q_far, num_seg=10):
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
    
    def getcost(self, q, minmax_q = [[-np.pi, np.pi],[-np.pi, np.pi]]):

        if minmax_q[0][0] < q.x < minmax_q[0][1] or minmax_q[1][0] < q.y < minmax_q[1][1]: # value must inside the range, not even equal pi
        
            ind_x = int((map_val(q.x, minmax_q[0][0], minmax_q[0][1], 0, self.map.shape[0])))
            ind_y = int((map_val(q.y, minmax_q[1][0], minmax_q[1][1], 0, self.map.shape[1])))

            return map[ind_y, ind_x] # in img format row is y, column is x, spend weeks cause of this

    def obstacle_cost(self, q_base, q_far, num_seg=10):
        seg = self.segmented_line(q_base, q_far, num_seg=num_seg)
        costlist = [self.getcost(q) for q in seg]

        if 0 in costlist:
            cost = 1e10
            return cost
        else:
            cost = np.sum(costlist) / num_seg
            return cost

    def line_cost(self, start: node, end: node) -> float:
        cost = self.w1 * (self.distance_cost(start, end) / (self.eta)) + self.w2 * (self.obstacle_cost(start, end))

        return cost

    def nearest(self, x_rand: node) -> node:
        vertex = []
        i = 0
        for x_near in self.nodes:
            dist = self.distance_cost(x_near, x_rand)
            vertex.append([dist, i, x_near])
            i += 1

        vertex.sort()
        x_nearest = vertex[0][2]

        return x_nearest

    def steer(self, x_rand: node, x_nearest: node) -> node:
        d = self.distance_cost(x_rand, x_nearest)

        if d < self.eta:
            x_new = node(x_rand.x, x_rand.y)
        else:
            new_x = x_nearest.x + self.eta * ((x_rand.x - x_nearest.x) / d)
            new_y = x_nearest.y + self.eta * ((x_rand.y - x_nearest.y) / d)

            x_new = node(new_x, new_y)

        return x_new

    def exist_check(self, x_new: node) -> bool:
        for x_near in self.nodes:
            if x_new.x == x_near.x and x_new.y == x_near.y:
                return False
            else:
                return True

    def new_check(self, x_new: node) -> bool:
        x_pob = np.array([x_new.x, x_new.y])
        x_pob = np.around(x_pob)

        if x_pob[0] >= self.map.shape[0]:
            x_pob[0] = self.map.shape[0] - 1
        if x_pob[1] >= self.map.shape[1]:
            x_pob[1] = self.map.shape[1] - 1

        x_pob = self.map[int(x_pob[1]), int(x_pob[0])]
        p = np.random.uniform(0, 1)

        if x_pob > p and self.exist_check(x_new):
            return True
        else:
            return False

    def add_parent(self, x_new: node, x_nearest: node) -> node:
        x_min = x_nearest
        c_min = x_min.cost + self.line_cost(x_min, x_new)

        for x_near in self.nodes:
            if self.distance_cost(x_near, x_new) <= self.eta:
                if x_near.cost + self.line_cost(x_near, x_new) < c_min:
                    x_min = x_near
                    c_min = x_near.cost + self.line_cost(x_near, x_new)
            x_new.parent = x_min
            x_new.cost = c_min

        return x_new

    def rewire(self, x_new: node):
        for x_near in self.nodes:
            if x_near is not x_new.parent:
                if (
                    self.distance_cost(x_near, x_new) <= self.eta
                ):  # and self.obstacle_cost(x_new, x_near) < 1
                    if x_new.cost + self.line_cost(x_new, x_near) < x_near.cost:
                        x_near.parent = x_new
                        x_near.cost = x_new.cost + self.line_cost(x_new, x_near)

    def get_path(self) -> list:
        temp_path = []
        path = []
        n = 0
        for i in self.nodes:
            if self.distance_cost(i, self.x_goal) < self.eta:  # 5
                cost = i.cost + self.line_cost(self.x_goal, i)
                temp_path.append([cost, n, i])
                n += 1
        temp_path.sort()

        if temp_path == []:
            print("cannot find path")
            return None

        else:
            closest_node = temp_path[0][2]
            i = closest_node
            self.x_goal.cost = temp_path[0][0]

            while i is not self.x_init:
                path.append(i)
                i = i.parent
            path.append(self.x_init)

            self.x_goal.parent = path[0]
            path.insert(0, self.x_goal)

            return path

    def draw_tree(self):
        for i in self.nodes:
            if i is not self.x_init:
                plt.plot([i.x, i.parent.x], [i.y, i.parent.y], "b")

    def draw_path(self, path: list):
        for i in path:
            if i is not self.x_init:
                plt.plot([i.x, i.parent.x], [i.y, i.parent.y], "r", linewidth=2.5)

    def start_planning(self):
        """start planning loop"""
        while True:
            # start record sampling time
            time_sampling_start = time.time()
            while True:
                # create random node by start uniform sampling
                x_rand = self.uniform_sampling()
                # update number of iteration
                self.total_iter += 1
                # find the nearest node to sampling
                x_nearest = self.nearest(x_rand)
                # create new node in the direction of random and nearest to nearest node
                x_new = self.steer(x_rand, x_nearest)
                # check whether to add new node to tree or not, if yes then add and breeak out of sampling loop, if not then continue sampling loop
                b = self.new_check(x_new)
                if b == True:
                    break
                # stop if the sampling iteration reach maximum
                if self.total_iter == self.maxiteration:
                    break
            # stop record sampling time
            time_sampling_end = time.time()
            # determine sampling time elapsed
            self.sampling_elapsed += time_sampling_end - time_sampling_start
            # stop the entire planner if iteration reach maximum
            if self.total_iter == self.maxiteration:
                break
            # update sample taken number
            self.sample_taken += 1
            print("==>> self.sample_taken: ", self.sample_taken)

            # start record addparent time
            time_addparent_start = time.time()
            # create xnew node with xnearest as its parent
            x_new = self.add_parent(x_new, x_nearest)
            # stop record addparent time
            time_addparent_end = time.time()
            # determine addparent time elapsed
            self.addparent_elapsed += time_addparent_end - time_addparent_start
            # add xnew to rrt tree nodes
            self.nodes.append(x_new)

            # start record rewire time
            time_rewire_start = time.time()
            # start rewire tree
            self.rewire(x_new)
            # stop record rewire time
            time_rewire_end = time.time()
            # determine rewire time elapsed
            self.rewire_elapsed += time_rewire_end - time_rewire_start

        # record end of planner loop
        self.e = time.time()

    def print_time(self):
        print("total time : ", self.e - self.s, "second")
        print(
            "sampling time : ",
            self.sampling_elapsed,
            "second",
            (self.sampling_elapsed * 100) / (self.e - self.s),
            "%",
        )
        print(
            "add_parent time : ",
            self.addparent_elapsed,
            "second",
            (self.addparent_elapsed * 100) / (self.e - self.s),
            "%",
        )
        print(
            "rewire time : ",
            self.rewire_elapsed,
            "second",
            (self.rewire_elapsed * 100) / (self.e - self.s),
            "%",
        )
        print("total_iteration = ", self.total_iter)
        print("cost : ", self.x_goal.cost)


if __name__ == "__main__":
    from map.map_loader import grid_map_binary

    np.random.seed(0)

    # SECTION - experiment 1
    map = grid_map_binary(index=1)
    plt.imshow(map)
    plt.show()
    x_init = np.array([0, 1]).reshape(2, 1)
    x_goal = np.array([2, 0]).reshape(2, 1)


    # SECTION - planner
    distance_weight = 0.5
    obstacle_weight = 0.5
    rrt = RobotRRTStarCostMapUniSampling(map, x_init, x_goal, distance_weight, obstacle_weight, maxiteration=1000)
    rrt.start_planning()
    # path = rrt.get_path()

    # SECTION - result
    plt.imshow(map)
    rrt.draw_tree()
    # rrt.draw_path(path)
    plt.show()
