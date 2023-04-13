import os
import sys

wd = os.path.abspath(os.getcwd())
sys.path.append(str(wd))

import time
import matplotlib.pyplot as plt
import numpy as np
from map.mapclass import map_val


class node:

    def __init__(self, x: float, y: float, cost: float = 0, parent=None):
        self.x = x
        self.y = y
        self.cost = cost
        self.parent = parent


class RobotRRTStarCostMapUniSampling:

    def __init__(self, mapclass, x_init: node, x_goal: node, w1: float, w2: float, eta: float = None, maxiteration: int = 1000):
        # map properties
        self.mapclass = mapclass
        self.x_min = self.mapclass.xmin
        self.x_max = self.mapclass.xmax
        self.y_min = self.mapclass.ymin
        self.y_max = self.mapclass.ymax
        self.x_init = node(x_init[0, 0], x_init[1, 0])
        self.x_goal = node(x_goal[0, 0], x_goal[1, 0])
        self.nodes = [self.x_init]

        # planner properties
        self.maxiteration = maxiteration
        self.eta = 0.3
        self.w1 = w1
        self.w2 = w2

        self.sample_taken = 0
        self.total_iter = 0
        self.graph_sample_num = 0
        self.graph_data = np.array([[0, 0]])

        self.s = time.time()
        self.e = None
        self.sampling_elapsed = 0
        self.addparent_elapsed = 0
        self.rewire_elapsed = 0

    def uniform_sampling(self) -> node:
        x = np.random.uniform(low=self.x_min, high=self.x_max)
        y = np.random.uniform(low=self.y_min, high=self.y_max)
        x_rand = node(x, y)
        return x_rand

    def distance_cost(self, start: node, end: node) -> float:
        distance_cost = np.linalg.norm([start.x - end.x, start.y - end.y])
        return distance_cost

    def segmented_line(self, q_base, q_far, num_seg=10):
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

    def getcost(self, q):
        if self.map.xmin < q.x < self.map.xmax and self.map.ymin < q.y < self.map.ymax:
            ind_x = int((map_val(q.x, self.map.xmin, self.map.xmax, 0, self.map.costmap.shape[0])))
            ind_y = int((map_val(q.y, self.map.ymin, self.map.ymax, 0, self.map.costmap.shape[1])))

            return self.map.costmap[ind_y, ind_x]

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

        if x_pob[0] >= self.map.costmap.shape[0]:
            x_pob[0] = self.map.costmap.shape[0] - 1
        if x_pob[1] >= self.map.costmap.shape[1]:
            x_pob[1] = self.map.costmap.shape[1] - 1

        x_pob = self.map.costmap[int(x_pob[1]), int(x_pob[0])]
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
                if self.distance_cost(x_near, x_new) <= self.eta:
                    if x_new.cost + self.line_cost(x_new, x_near) < x_near.cost:
                        x_near.parent = x_new
                        x_near.cost = x_new.cost + self.line_cost(x_new, x_near)

    def get_path(self) -> list:
        temp_path = []
        path = []
        n = 0
        for i in self.nodes:
            if self.distance_cost(i, self.x_goal) < self.eta:
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
            while True:
                x_rand = self.uniform_sampling()
                self.total_iter += 1
                x_nearest = self.nearest(x_rand)
                x_new = self.steer(x_rand, x_nearest)
                b = self.new_check(x_new)
                if b == True:
                    break
                if self.total_iter == self.maxiteration:
                    break
            if self.total_iter == self.maxiteration:
                break
            self.sample_taken += 1
            print("==>> self.sample_taken: ", self.sample_taken)

            x_new = self.add_parent(x_new, x_nearest)
            self.nodes.append(x_new)

            self.rewire(x_new)


if __name__ == "__main__":
    from map.mapclass import MapClass, MapLoader
    np.random.seed(0)

    maploader = MapLoader.loadsave(maptype="task", mapindex=1, reverse=False)
    mapclass = MapClass(maploader, maprange=[[-np.pi, np.pi], [-np.pi, np.pi]])
    plt.imshow(map.costmap)
    plt.show()
    x_init = np.array([0, 1]).reshape(2, 1)
    x_goal = np.array([2, 0]).reshape(2, 1)

    distance_weight = 0.5
    obstacle_weight = 0.5
    rrt = RobotRRTStarCostMapUniSampling(map, x_init, x_goal, distance_weight, obstacle_weight, maxiteration=1000)
    rrt.start_planning()

    plt.imshow(map.costmap)
    rrt.draw_tree()

    plt.show()
