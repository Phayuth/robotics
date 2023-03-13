import numpy as np
from shapely.geometry import Point
import matplotlib.pyplot as plt
import time

class node(object):
    def __init__(self, x, y, cost = 0, parent = None ):

        self.x = x
        self.y = y
        self.arr = np.array([self.x, self.y])
        self.cost = cost
        self.parent = parent

class rrt_general():
    def __init__(self, map, x_init, x_goal, eta, obs, obstacle_center, collision_range, max_iteration):

        self.map = map
        self.obs = obs
        self.obstacle_center = obstacle_center
        self.collision_range = collision_range
        self.map_size = self.map.shape[0]
        self.eta = eta
        self.x_init = x_init
        self.x_goal = x_goal
        self.nodes = [self.x_init]

        self.iteration = max_iteration
        self.total_iter = 0
        self.sample_taken = 0

        self.s = time.time()
        self.e = None
        self.t1, self.t2, self.t3 = 0, 0, 0

    def Sampling(self):

        x = np.random.uniform(low=0, high=self.map_size)
        y = np.random.uniform(low=0, high=self.map_size)

        x_rand = node(x, y)

        return x_rand

    def Node_collision_check(self, sample):

        p = Point(sample.x, sample.y)

        for o in self.obs:
            if p.within(o):
                return False
        return True

    def Edge_collision_check(self,x_nearest, x_new):

        for c in self.obstacle_center:

            AC = c - x_nearest.arr
            AB = x_new.arr - x_nearest.arr
            d = np.linalg.norm(x_new.arr - x_nearest.arr)
            v = np.dot(AB,AC) / d

            if v < 0 :
                r = np.linalg.norm(c - x_nearest.arr)
            elif v > d :
                r = np.linalg.norm(c - x_new.arr)
            else:
                area = abs(AB[0]*AC[1] - AC[0]*AB[1]) / 2
                r = area/d
            # num = 0

            if r <= self.collision_range: #self.collision_range[num]:

                return False
            # num+=1
        return True

    def Distance_Cost(self, start, end):

        distance_cost = np.linalg.norm(start.arr - end.arr)

        return distance_cost

    def Line_Cost(self, start, end):

        cost = self.Distance_Cost(start, end)/(self.eta)

        return cost

    def Nearest(self, x_rand):

        vertex = []
        v = []
        i = 0
        for x_near in self.nodes:

            dist = self.Distance_Cost(x_near, x_rand)
            vertex.append([dist, i, x_near])
            i+=1

        vertex.sort()
        x_nearest = vertex[0][2]

        return x_nearest

    def Steer(self, x_rand, x_nearest):

        d = self.Distance_Cost(x_rand, x_nearest)

        if d < self.eta :
            x_new = node(x_rand.x, x_rand.y)
        else:
            new_x = x_nearest.x + self.eta * ((x_rand.x - x_nearest.x)/d)
            new_y = x_nearest.y + self.eta * ((x_rand.y - x_nearest.y)/d)

            x_new  = node(new_x, new_y)

        return x_new

    def Get_Path(self):

        temp_path = []
        path = []
        n = 0
        for i in self.nodes:
            if self.Distance_Cost(i, self.x_goal) <= 3:
                cost = i.cost + self.Line_Cost(self.x_goal, i)
                temp_path.append([cost,n, i])
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

    def Draw_Tree(self):

        for i in self.nodes:
            if i is not self.x_init:
                plt.plot([i.x, i.parent.x], [i.y, i.parent.y], "blue")

    def Draw_obs(self):

        fig, axs = plt.subplots(figsize=(10, 10))
        for o in self.obs:

            x, y = o.exterior.xy
            axs.fill(x, y, fc="black", ec="none")

    def Draw_path(self, path):

        for i in path:
            if i is not self.x_init:
                plt.plot([i.x, i.parent.x], [i.y, i.parent.y], "r", linewidth=2.5)

    def Cost_Graph(self):

        temp_path = []
        path = []
        n = 0
        for i in self.nodes:
            if self.Distance_Cost(i, self.x_goal) < 5:
                cost = i.cost + self.Line_Cost(self.x_goal, i)
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

    def start_planning(self):
        while True:

            s1 = time.time()
            while True:

                x_rand = self.Sampling()

                self.total_iter += 1
                if self.total_iter % 100 == 0:
                    print(self.total_iter, "Iteration")

                x_nearest = self.Nearest(x_rand)

                x_new = self.Steer(x_rand, x_nearest)

                if self.Node_collision_check(x_new):
                    break

                if self.total_iter > self.iteration:
                    break

            if self.total_iter > self.iteration:
                break

            e1 = time.time()
            self.t1 += (e1 - s1)

            if self.Edge_collision_check(x_nearest, x_new) == False:
                continue

            self.sample_taken += 1

            x_new.cost = x_nearest.cost + self.Line_Cost(x_nearest, x_new)
            x_new.parent = x_nearest

            self.nodes.append(x_new)

        self.e = time.time()

    def print_time(self):
        print("Total time : ", self.e - self.s,"초")
        print("Sampling time : ", self.t1,"초", (self.t1*100)/(self.e-self.s),"%")
        print("Total_sample = ", self.sample_taken)
        print("Cost : ", self.x_goal.cost)