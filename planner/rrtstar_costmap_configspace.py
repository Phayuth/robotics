import os
import sys

wd = os.path.abspath(os.getcwd())
sys.path.append(str(wd))

import numpy as np
import matplotlib.pyplot as plt
import time
from map.mapclass import map_val


class node(object):

    def __init__(self, x: float, y: float, cost: float = 0, parent=None):
        self.x = x
        self.y = y
        self.cost = cost
        self.parent = parent


class RrtstarCostmapConfigspace():

    def __init__(self, mapclass, x_start: node, x_goal: node, distance_weight: float, obstacle_weight: float, eta: float = None, maxiteration: int = 1000):
        # map properties
        self.mapclass = mapclass
        self.costmap = self.mapclass.costmap
        self.x_min = self.mapclass.xmin
        self.x_max = self.mapclass.xmax
        self.y_min = self.mapclass.ymin
        self.y_max = self.mapclass.ymax
        self.x_start = node(x_start[0, 0], x_start[1, 0])
        self.x_start.cost = 0.0
        self.x_goal = node(x_goal[0, 0], x_goal[1, 0])

        # planner properties
        self.maxiteration = maxiteration
        self.m = (self.x_max - self.x_min) * (self.y_max - self.y_min)
        self.r = (2 * (1 + 1/2)**(1 / 2)) * (self.m / np.pi)**(1 / 2)
        self.eta = self.r * (np.log(self.maxiteration) / self.maxiteration)**(1 / 2)
        self.w1 = distance_weight
        self.w2 = obstacle_weight
        self.nodes = [self.x_start]

        self.sample_taken = 0
        self.total_iter = 0

        # timing
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

    def bias_sampling(self) -> node:
        row = self.costmap.shape[1]
        p = np.ravel(self.costmap) / np.sum(self.costmap)
        x_sample = np.random.choice(len(p), p=p)
        x = x_sample // row
        y = x_sample % row
        x = np.random.uniform(low=x, high=x)
        y = np.random.uniform(low=y, high=y)
        x = map_val(x, 0, self.costmap.shape[1], self.x_min, self.x_max)
        y = map_val(y, 0, self.costmap.shape[0], self.y_min, self.y_max)
        x_rand = node(x, y)
        return x_rand

    def distance_cost(self, start: node, end: node) -> float:
        distance_cost = np.linalg.norm([(start.x - end.x), (start.y - end.y)])
        return distance_cost

    def obstacle_cost(self, start: float, end: float) -> float:
        seg_length = (self.x_max - self.x_min) / self.costmap.shape[0]
        seg_point = int(np.ceil(self.distance_cost(start, end) / seg_length))

        costv = []
        if seg_point > 1:
            v = np.array([end.x - start.x, end.y - start.y]) / (seg_point)

            for i in range(seg_point + 1):
                seg = np.array([start.x, start.y]) + i*v
                costindex_x = map_val(seg[1], self.x_min, self.x_max, 0, self.costmap.shape[1])
                costindex_y = map_val(seg[0], self.y_min, self.y_max, 0, self.costmap.shape[0])

                if cost := 1 - self.costmap[int(costindex_x), int(costindex_y)] == 1:
                    cost = 1e10

                costv.append(cost)

            cost = sum(costv) / (seg_point+1)
            return cost

        else:
            start_costindex_x = map_val(start.x, self.x_min, self.x_max, 0, self.costmap.shape[1])
            start_costindex_y = map_val(start.y, self.y_min, self.y_max, 0, self.costmap.shape[0])

            end_costindex_x = map_val(end.x, self.x_min, self.x_max, 0, self.costmap.shape[1])
            end_costindex_y = map_val(end.y, self.y_min, self.y_max, 0, self.costmap.shape[0])

            cost = ((self.costmap[int(start_costindex_y), int(start_costindex_x)] + self.costmap[int(end_costindex_y), int(end_costindex_x)])) / 2
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
        costindex_x = map_val(x_new.x, self.x_min, self.x_max, 0, self.costmap.shape[1])
        costindex_y = map_val(x_new.y, self.y_min, self.y_max, 0, self.costmap.shape[0])
        x_pob = np.array([costindex_x, costindex_y])
        x_pob = np.around(x_pob)

        if x_pob[0] >= self.costmap.shape[0]:
            x_pob[0] = self.costmap.shape[0] - 1
        if x_pob[1] >= self.costmap.shape[1]:
            x_pob[1] = self.costmap.shape[1] - 1

        x_pob = self.costmap[int(x_pob[1]), int(x_pob[0])]
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
            if self.distance_cost(i, self.x_goal) < self.eta:  #5
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

            while i is not self.x_start:
                path.append(i)
                i = i.parent
            path.append(self.x_start)

            self.x_goal.parent = path[0]
            path.insert(0, self.x_goal)

            return path

    def plt_env(self):
        obs = self.mapclass.costmap2geo(free_space_value=0)
        for obs in obs:
            obs.plot()

        for i in self.nodes:
            if i is not self.x_start:
                plt.plot([i.x, i.parent.x], [i.y, i.parent.y], "b")

    def draw_path(self, path: list):
        for i in path:
            if i is not self.x_start:
                plt.plot([i.x, i.parent.x], [i.y, i.parent.y], "r", linewidth=2.5)

    def planning(self):
        while True:
            time_sampling_start = time.time()
            while True:
                x_rand = self.bias_sampling()
                self.total_iter += 1
                x_nearest = self.nearest(x_rand)
                x_new = self.steer(x_rand, x_nearest)
                b = self.new_check(x_new)
                if b:
                    break
                if self.total_iter == self.maxiteration:
                    break
            time_sampling_end = time.time()
            self.sampling_elapsed += time_sampling_end - time_sampling_start
            if self.total_iter == self.maxiteration:
                break
            self.sample_taken += 1
            print("==>> self.sample_taken: ", self.sample_taken)

            time_addparent_start = time.time()
            x_new = self.add_parent(x_new, x_nearest)
            time_addparent_end = time.time()
            self.addparent_elapsed += (time_addparent_end - time_addparent_start)
            self.nodes.append(x_new)

            time_rewire_start = time.time()
            self.rewire(x_new)
            time_rewire_end = time.time()
            self.rewire_elapsed += (time_rewire_end - time_rewire_start)

        self.e = time.time()

    def print_time(self):
        print("total time : ", self.e - self.s, "second")
        print("sampling time : ", self.sampling_elapsed, "second", (self.sampling_elapsed * 100) / (self.e - self.s), "%")
        print("add_parent time : ", self.addparent_elapsed, "second", (self.addparent_elapsed * 100) / (self.e - self.s), "%")
        print("rewire time : ", self.rewire_elapsed, "second", (self.rewire_elapsed * 100) / (self.e - self.s), "%")
        print("total_iteration = ", self.total_iter)
        print("cost : ", self.x_goal.cost)


if __name__ == "__main__":
    from map.mapclass import CostMapLoader, CostMapClass
    np.random.seed(1)
    from util.extract_path_class import extract_path_class_2d

    # SECTION - Experiment 1
    maploader = CostMapLoader.loadsave(maptype="task", mapindex=2)
    maploader.grid_map_probability(size=3)
    mapclass = CostMapClass(maploader=maploader, maprange=[[-np.pi, np.pi], [-np.pi, np.pi]])
    plt.imshow(mapclass.costmap, origin="lower")
    plt.show()
    x_start = np.array([-2.5, -2.5]).reshape(2, 1)
    x_goal = np.array([2.5, 2.5]).reshape(2, 1)

    # SECTION - Experiment 2
    # maper = np.ones((100,100))
    # maploader = CostMapLoader.loadarray(costmap=maper)
    # mapclass = CostMapClass(maploader=maploader, maprange=[[-np.pi, np.pi], [-np.pi, np.pi]])
    # plt.imshow(mapclass.costmap)
    # plt.show()
    # x_start = np.array([0,0]).reshape(2, 1)
    # x_goal = np.array([1,1]).reshape(2, 1)

    # SECTION - planner
    distance_weight = 0.5
    obstacle_weight = 0.5
    rrt = RrtstarCostmapUnisampling(mapclass, x_start, x_goal, distance_weight, obstacle_weight, maxiteration=1000)
    rrt.planning()
    path = rrt.get_path()

    px, py = extract_path_class_2d(path)
    print(f"==>> px: \n{px}")
    print(f"==>> py: \n{py}")
    # SECTION - result
    rrt.plt_env()
    rrt.draw_path(path)
    plt.show()