import numpy as np
from shapely.geometry import Point
import matplotlib.pyplot as plt
import time

class node(object):
    def __init__(self, x:float, y:float, cost:float = 0, parent = None):
        """node class

        Args:
            x (float): configuration x
            y (float): configuration y
            cost (float, optional): cost value. Defaults to 0.
            parent (numpy array, optional): parent node. Defaults to None.
        """
        self.x = x
        self.y = y
        self.arr = np.array([self.x, self.y])
        self.cost = cost
        self.parent = parent

class rrt_star():
    def __init__(self, map, x_init, x_goal, eta, obs, obstacle_center, collision_range, map_size, max_iteration):

        self.map = map
        self.x_init = x_init
        self.x_goal = x_goal

        self.obs = obs
        self.obstacle_center = obstacle_center
        self.collision_range = collision_range
        self.map_size = map_size
        self.eta = eta
        
        self.nodes = [self.x_init]

        self.iteration = max_iteration
        self.sample_taken = 0
        self.total_iter = 0

        self.t1, self.t2, self.t3 = 0, 0, 0

    def Sampling(self):

        x = np.random.uniform(low=self.map_size[0], high=self.map_size[1])
        y = np.random.uniform(low=self.map_size[0], high=self.map_size[1])

        x_rand = node(x, y)

        return x_rand

    # def Bias_sampling(self):
    #     row = self.map.shape[1]

    #     p = np.ravel(self.map) / np.sum(self.map)

    #     x_sample = np.random.choice(len(p), p=p)

    #     x = x_sample // row
    #     y = x_sample % row

    #     x = np.random.uniform(low=x - 0.5, high=x + 0.5)
    #     y = np.random.uniform(low=y - 0.5, high=y + 0.5)

    #     x_rand = node(x, y)

    #     return x_rand

    def New_Check(self, x_new):

        x_pob = np.array([x_new.x, x_new.y])
        x_pob = np.around(x_pob)

        if x_pob[0] >= self.map.shape[0]:
            x_pob[0] = self.map.shape[0] - 1
        if x_pob[1] >= self.map.shape[1]:
            x_pob[1] = self.map.shape[1] - 1

        x_pob = self.map[int(x_pob[0]), int(x_pob[1])]
        p = np.random.uniform(0, 1)
        # print(x_pob,p)
        if x_pob > p and self.Exist_Check(x_new):
            return True
        else:
            return False

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

    def Exist_Check(self, x_new):

        for x_near in self.nodes:
            if x_new.x == x_near.x and x_new.y == x_near.y :
                return False
            else :
                return True

    def Add_Parent(self, x_new, x_nearest):

        x_new.cost = x_nearest.cost + self.Line_Cost(x_nearest, x_new)
        x_new.parent = x_nearest
        for x_near in self.nodes:
            if self.Distance_Cost(x_near, x_new) <= self.eta and self.Edge_collision_check(x_near, x_new) :
                if  x_near.cost + self.Line_Cost(x_near, x_new) < x_nearest.cost + self.Line_Cost(x_nearest, x_new):

                    x_nearest = x_near
                    x_new.cost = x_nearest.cost + self.Line_Cost(x_nearest, x_new)
                    x_new.parent = x_nearest

        return x_new, x_nearest

    def Rewire(self, x_new):

        for x_near in self.nodes:
            if x_near is not x_new.parent:
                if self.Distance_Cost(x_near, x_new) <= self.eta and self.Edge_collision_check(x_new, x_near) :
                    if x_new.cost + self.Line_Cost(x_new, x_near) < x_near.cost:

                        x_near.parent = x_new
                        x_near.cost = x_new.cost + self.Line_Cost(x_new, x_near)

        return

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

                if self.total_iter == self.iteration:
                    break
            e1 = time.time()
            self.t1 += e1- s1

            if self.total_iter == self.iteration:
                break

            s2 = time.time()
            if self.Edge_collision_check(x_nearest, x_new) == False:
                continue

            self.sample_taken += 1

            x_new, x_nearest = self.Add_Parent(x_new, x_nearest)
            e2 = time.time()
            self.t2 += e2 - s2

            self.nodes.append(x_new)

            s3 = time.time()
            self.Rewire(x_new)
            e3 = time.time()
            self.t3 += e3 - s3

            e = time.time()

    def print_time(self):
        # print("Sampling time : ", self.t1/(iter),"초", (self.t1*100)/total_time,"%")
        # print("Add_Parent time : ", t2/ (iter),"초", (t2*100)/total_time,"%")
        # print("Rewire time : ", t3/ (iter),"초", (t3*100)/total_time,"%")
        # print("Total_time : ", total_time / (iter) , np.std(iter_time))
        # print("total_sample : ", total_sample / (iter), np.std(sample))
        # print("total_cost : ", total_cost / (iter-count), np.std(cost))
        pass