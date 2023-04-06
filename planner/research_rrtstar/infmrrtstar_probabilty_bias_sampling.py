import os
import sys

wd = os.path.abspath(os.getcwd())
sys.path.append(str(wd))

import numpy as np
import matplotlib.pyplot as plt
import time


class node(object):
    def __init__(self, x: float, y: float, cost: float = 0, costr:float = 0, parent=None):
        self.x = x
        self.y = y
        self.cost = cost
        self.costr = costr
        self.parent = parent


class infmrrtstar_binarymap_biassampling:
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
        self.m = map.shape[0] * map.shape[1]
        self.r = (2 * (1 + 1 / 2) ** (1 / 2)) * (self.m / np.pi) ** (1 / 2)
        self.eta = self.r * (np.log(self.maxiteration) / self.maxiteration) ** (1 / 2)
        self.w1 = w1
        self.w2 = w2
        self.X_soln = []

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

    def bias_sampling(self)-> node:
        row = self.map.shape[1]
        p = np.ravel(self.map) / np.sum(self.map)
        x_sample = np.random.choice(len(p), p=p)
        x = x_sample // row
        y = x_sample % row
        x = np.random.uniform(low=x - 0.5, high=x + 0.5)
        y = np.random.uniform(low=y - 0.5, high=y + 0.5)
        x_rand = node(x, y)
        return x_rand
    
    def sampling(self, x_start, x_goal, c_max):
        if c_max < np.inf:
            c_min = self.distance_cost(x_start, x_goal)
            print(c_max, c_min)
            x_center = np.array([(x_start.x + x_goal.x)/2, (x_start.y + x_goal.y)/2]).reshape(2,1)
            C = self.RotationToWorldFrame(x_start, x_goal)
            r1 = c_max/2
            r2 = np.sqrt(c_max**2 - c_min**2)/2
            L = np.diag([r1, r2])
            while True:
                x_ball = self.sampleUnitBall()
                x_rand = (C @ L @ x_ball) + x_center
                x_rand = node(x_rand[0,0], x_rand[1,0])
                if (0 < x_rand.x < self.map.shape[0] -1) and (0 < x_rand.y < self.map.shape[1] -1): # check if inside configspace
                    break
        else:
            x_rand = self.bias_sampling()
        return x_rand

    def sampleUnitBall(self):
        r = np.random.uniform(low=0, high=1)
        theta = np.random.uniform(low=0, high=2*np.pi)
        x = r*np.cos(theta)
        y = r*np.sin(theta)
        return np.array([[x], [y]]).reshape(2,1)

    def ingoal_region(self, x_new):
        if np.linalg.norm([self.x_goal.x - x_new.x, self.x_goal.y - x_new.y]) <= 50 :# self.eta:
            return True
        else:
            return False
        
    def RotationToWorldFrame(self, x_start, x_goal):
        theta = np.arctan2((x_goal.y - x_start.y), (x_goal.x - x_start.x))

        R = np.array([[ np.cos(theta), np.sin(theta)],
                      [-np.sin(theta), np.cos(theta)]]).T
        
        return R
        
    def distance_cost(self, start: node, end: node) -> float:

        distance_cost = np.linalg.norm([start.x - end.x, start.y - end.y])

        return distance_cost

    def obstacle_cost(self, start: float, end: float) -> float:

        seg_length = 1
        seg_point = int(np.ceil(self.distance_cost(start, end) / seg_length))

        value = 0
        if seg_point > 1:
            v = np.array([end.x - start.x, end.y - start.y]) / (seg_point)

            for i in range(seg_point + 1):
                seg = np.array([start.x, start.y]) + i * v
                seg = np.around(seg)
                if 1 - self.map[int(seg[1]), int(seg[0])] == 1:
                    cost = 1e10

                    return cost
                else:
                    value += 1 - self.map[int(seg[1]), int(seg[0])]

            cost = value / (seg_point + 1)

            return cost

        else:

            value = (
                self.map[int(start.y), int(start.x)] + self.map[int(end.y), int(end.x)]
            )
            cost = value / 2

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
        c_minr = x_min.costr + self.distance_cost(x_min, x_new) # real cost

        for x_near in self.nodes:
            if self.distance_cost(x_near, x_new) <= self.eta:
                if x_near.cost + self.line_cost(x_near, x_new) < c_min:
                    x_min = x_near
                    c_min = x_near.cost + self.line_cost(x_near, x_new)
                    c_minr = x_near.costr + self.distance_cost(x_near, x_new)
            x_new.parent = x_min
            x_new.cost = c_min
            x_new.costr = c_minr

        return x_new

    def rewire(self, x_new: node):

        for x_near in self.nodes:
            if x_near is not x_new.parent:
                if (self.distance_cost(x_near, x_new) <= self.eta):  # and self.obstacle_cost(x_new, x_near) < 1
                    if x_new.cost + self.line_cost(x_new, x_near) < x_near.cost:
                        x_near.parent = x_new
                        x_near.cost = x_new.cost + self.line_cost(x_new, x_near)
                        x_near.costr = x_new.costr + self.distance_cost(x_new, x_near)

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

    def cost_graph(self):

        temp_path = []
        n = 0
        for i in self.nodes:
            if self.distance_cost(i, self.x_goal) < 3:
                cost = i.cost + self.line_cost(self.x_goal, i)
                temp_path.append([cost, n, i])
                n += 1
        temp_path.sort()

        if temp_path == []:
            return 0
        else:
            closest_node = temp_path[0][2]
            i = closest_node

            self.x_goal.cost = temp_path[0][0]

            return self.x_goal.cost

    def draw_tree(self):

        for i in self.nodes:
            if i is not self.x_init:
                plt.plot([i.x, i.parent.x], [i.y, i.parent.y], "b")

    def draw_path(self, path: list):

        for i in path:
            if i is not self.x_init:
                plt.plot([i.x, i.parent.x], [i.y, i.parent.y], "r", linewidth=2.5)

    def check_can_connect_to_goal(self, path_iter):

        x_nearest = self.nearest(self.x_goal)
        if self.distance_cost(x_nearest, self.x_goal) < 5:

            path_iter += 1

            return path_iter

        else:
            return path_iter

    def start_planning(self):
        while True:
            c_best = np.inf
            while True:
                for x_soln in self.X_soln:
                    c_best = x_soln.parent.costr + self.distance_cost(x_soln.parent, x_soln) + self.distance_cost(x_soln, self.x_goal)
                    if x_soln.parent.costr + self.distance_cost(x_soln.parent, x_soln) + self.distance_cost(x_soln, self.x_goal) < c_best:
                        c_best = x_soln.parent.costr + self.distance_cost(x_soln.parent, x_soln) + self.distance_cost(x_soln, self.x_goal)

                x_rand = self.sampling(self.x_init, self.x_goal, c_best)
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

            # in goal region
            if self.ingoal_region(x_new):
                self.X_soln.append(x_new)


            # record graph cost data
            # self.Graph_sample_num += 1
            # if self.Graph_sample_num % 100 == 0:
            #     Graph_cost = self.Cost_Graph()
            #     self.Graph_data = np.append(self.Graph_data,np.array([[self.Graph_sample_num, Graph_cost]]), axis = 0)

        # record end of planner loop
        self.e = time.time()

    def print_time(self):
        print("total time : ", self.e - self.s, "second")
        print("sampling time : ",self.sampling_elapsed,"second",(self.sampling_elapsed * 100) / (self.e - self.s),"%")
        print("add_parent time : ",self.addparent_elapsed,"second",(self.addparent_elapsed * 100) / (self.e - self.s),"%")
        print("rewire time : ",self.rewire_elapsed,"second",(self.rewire_elapsed * 100) / (self.e - self.s),"%")
        print("total_iteration = ", self.total_iter)
        print("cost : ", self.x_goal.cost)


if __name__ == "__main__":
    from map.map_loader import grid_map_binary
    np.random.seed(0)

    # SECTION - Experiment 1
    # map = grid_map_binary(index=1)
    # plt.imshow(map)
    # plt.show()
    # x_init = np.array([24, 12]).reshape(2, 1)
    # x_goal = np.array([1.20, 13.20]).reshape(2, 1)


    # SECTION - Experiment 2
    # map = np.ones((500,500))
    # plt.imshow(map)
    # plt.show()
    # x_init = np.array([20, 20]).reshape(2, 1)
    # x_goal = np.array([200, 20]).reshape(2, 1)


    # SECTION - Experiment 3
    map = grid_map_binary(index=1)
    plt.imshow(map)
    plt.show()
    x_init = np.array([5, 5]).reshape(2, 1)
    x_goal = np.array([25, 5]).reshape(2, 1)


    # SECTION - planner
    distance_weight = 0.5
    obstacle_weight = 0.5
    rrt = infmrrtstar_binarymap_biassampling(map, x_init, x_goal, distance_weight, obstacle_weight, maxiteration=500)
    rrt.start_planning()
    path = rrt.get_path()


    # SECTION - result
    plt.imshow(map)
    rrt.draw_tree()
    rrt.draw_path(path)
    plt.show()
