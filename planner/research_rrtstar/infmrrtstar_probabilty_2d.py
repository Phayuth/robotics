import os
import sys
wd = os.path.abspath(os.getcwd())
sys.path.append(str(wd))

import numpy as np
import matplotlib.pyplot as plt
import time

class node(object):
    def __init__(self, x:float, y:float, cost:float = 0, parent = None):
        self.x = x
        self.y = y
        self.arr = np.array([self.x, self.y])
        self.cost = cost
        self.parent = parent

class rrt_star():
    def __init__(self, map:np.ndarray, x_init:node, x_goal:node, w1:float, w2:float, eta:float=None, maxiteration:int=1000):
        # map properties
        self.map = map
        self.x_init = node(x_init[0,0], x_init[1,0])
        self.x_goal = node(x_goal[0,0], x_goal[1,0])
        self.nodes = [self.x_init]
        
        # planner properties
        self.maxiteration = maxiteration
        self.m = map.shape[0] * map.shape[1]
        self.r = (2 * (1 + 1/2)**(1/2)) * (self.m/np.pi)**(1/2)
        self.eta = self.r * (np.log(self.maxiteration) / self.maxiteration)**(1/2)
        self.w1 = w1
        self.w2 = w2
        self.X_soln = []

        self.sample_taken = 0
        self.total_iter = 0
        self.Graph_sample_num = 0
        self.Graph_data = np.array([[0,0]])

        # timing
        self.s = time.time()
        self.e = None
        self.sampling_elapsed = 0
        self.addparent_elapsed = 0
        self.rewire_elapsed = 0

    def biassampling(self)-> node:
        row = self.map.shape[1]
        p = np.ravel(self.map) / np.sum(self.map)
        x_sample = np.random.choice(len(p), p=p)
        x = x_sample // row
        y = x_sample % row
        x = np.random.uniform(low=x - 0.5, high=x + 0.5)
        y = np.random.uniform(low=y - 0.5, high=y + 0.5)
        x_rand = node(x, y)
        return x_rand
    
    def infmsampling(self, x_start, x_goal, c_max):
        if c_max < np.inf:
            c_min = self.Distance_cost(x_start, x_goal)
            x_center = np.array([(x_start.x + x_goal.x)/2, (x_start.y + x_goal.y)/2]).reshape(2,1)
            C = self.RotationToWorldFrame(x_start, x_goal)
            r1 = c_max/2
            r2 = np.sqrt(c_max**2 - c_min**2)/2
            L = np.diag([r1, r2])
            x_ball = self.sampleUnitBall()
            x_rand = (C @ L @ x_ball) + x_center
            x_rand = node(x_rand[0,0], x_rand[1,0])
        else:
            x_rand = self.biassampling()
        return x_rand

    def sampleUnitBall(self):
        r = np.random.uniform(low=0, high=1)
        theta = np.random.uniform(low=0, high=2*np.pi)
        x = r*np.cos(theta)
        y = r*np.sin(theta)
        return np.array([[x], [y]]).reshape(2,1)
    
    def RotationToWorldFrame(self, x_start, x_goal):
        theta = np.arctan2((x_goal.y - x_start.y), (x_goal.x - x_start.x))

        R = np.array([[ np.cos(theta), np.sin(theta)],
                      [-np.sin(theta), np.cos(theta)]]).T
        
        return R
    
    def ingoal_region(self, x_new):
        if np.linalg.norm([self.x_goal.x - x_new.x, self.x_goal.y - x_new.y]) <= 1 :# self.eta:
            return True
        else:
            return False

    def Distance_cost(self, start:node, end:node)-> float:
        distance_cost = np.linalg.norm(start.arr - end.arr)
        return distance_cost

    def Obstacle_cost(self, start:node, end:node)-> float:
        seg_length = 1
        seg_point = int(np.ceil(np.linalg.norm(start.arr - end.arr) / seg_length))

        value = 0
        if seg_point > 1:
            v = (end.arr - start.arr) / (seg_point)

            for i in range(seg_point + 1):
                seg = start.arr + i * v
                seg = np.around(seg)
                if 1 - self.map[int(seg[0]), int(seg[1])] == 1 :
                    cost = 1e10

                    return cost
                else:
                    value += 1 - self.map[int(seg[0]), int(seg[1])]

            cost = value / (seg_point + 1)

            return cost

        else:

            value = self.map[int(start.arr[0]), int(start.arr[1])] + self.map[int(end.arr[0]), int(end.arr[1])]
            cost = value / 2

            return cost

    def Line_cost(self, start:node, end:node)-> float:
        cost = self.w1*(self.Distance_cost(start, end)/self.eta) + self.w2*(self.Obstacle_cost(start, end))
        return cost

    def Nearest(self, x_rand:node)-> node:
        vertex = []
        i = 0
        for x_near in self.nodes:

            dist = self.Distance_cost(x_near, x_rand)
            vertex.append([dist, i, x_near])
            i+=1

        vertex.sort()
        x_nearest = vertex[0][2]

        return x_nearest

    def Steer(self, x_rand:node, x_nearest:node)-> node:
        d = self.Distance_cost(x_rand, x_nearest)

        if d < self.eta :
            x_new = node(x_rand.x, x_rand.y)
        else:
            new_x = x_nearest.x + self.eta*((x_rand.x - x_nearest.x)/d)
            new_y = x_nearest.y + self.eta*((x_rand.y - x_nearest.y)/d)

            x_new  = node(new_x, new_y)

        return x_new

    def Exist_Check(self, x_new:node)-> bool:
        for x_near in self.nodes:
            if x_new.x == x_near.x and x_new.y == x_near.y:
                return False
            else :
                return True

    def New_Check(self, x_new:node)-> bool:
        x_pob = np.array([x_new.x, x_new.y])
        x_pob = np.around(x_pob)

        if x_pob[0] >= self.map.shape[0]:
            x_pob[0] = self.map.shape[0] - 1
        if x_pob[1] >= self.map.shape[1]:
            x_pob[1] = self.map.shape[1] - 1

        x_pob = self.map[int(x_pob[0]), int(x_pob[1])]
        p = np.random.uniform(0, 1)

        if x_pob > p and self.Exist_Check(x_new):
            return True
        else:
            return False

    def Add_Parent(self, x_new:node, x_nearest:node)-> node:
        x_min = x_nearest
        c_min = x_min.cost + self.Line_cost(x_min, x_new)

        for x_near in self.nodes:
            if self.Distance_cost(x_near, x_new) <= self.eta :
                if x_near.cost + self.Line_cost(x_near, x_new) < c_min:
                    x_min = x_near
                    c_min = x_near.cost + self.Line_cost(x_near, x_new)
            x_new.parent = x_min
            x_new.cost = c_min

        return x_new

    def Rewire(self, x_new:node):
        for x_near in self.nodes:
            if x_near is not x_new.parent:
                if self.Distance_cost(x_near, x_new) <= self.eta:
                    if x_new.cost + self.Line_cost(x_new, x_near) < x_near.cost:
                        x_near.parent = x_new
                        x_near.cost = x_new.cost + self.Line_cost(x_new, x_near)

    def Get_Path(self)-> list:
        temp_path = []
        path = []
        n = 0
        for i in self.nodes:
            if self.Distance_cost(i, self.x_goal) < self.eta: #5
                cost = i.cost + self.Line_cost(self.x_goal, i)
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

    def Cost_Graph(self):

        temp_path = []
        n = 0
        for i in self.nodes:
            if self.Distance_cost(i, self.x_goal) < 3:
                cost = i.cost + self.Line_cost(self.x_goal, i)
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

    def Draw_Tree(self):
        for i in self.nodes:
            if i is not self.x_init:
                plt.plot([i.x, i.parent.x], [i.y, i.parent.y], "b")

    def Draw_path(self, path:list):
        for i in path:
            if i is not self.x_init:
                plt.plot([i.x, i.parent.x], [i.y, i.parent.y], "r", linewidth=2.5)

    def start_planning(self):
        time_sampling_start = time.time()
        c_best = np.inf
        for itera in range(self.maxiteration):
            print(c_best)
            for x_soln in self.X_soln:
                c_best = x_soln.parent.cost + self.Distance_cost(x_soln.parent, x_soln) + self.Distance_cost(x_soln, self.x_goal)
                if x_soln.parent.cost + self.Distance_cost(x_soln.parent, x_soln) + self.Distance_cost(x_soln, self.x_goal) < c_best:
                    c_best = x_soln.parent.cost + self.Distance_cost(x_soln.parent, x_soln) + self.Distance_cost(x_soln, self.x_goal)
            
            x_rand = self.infmsampling(self.x_init, self.x_goal, c_best)

            x_nearest = self.Nearest(x_rand)
            x_new = self.Steer(x_rand, x_nearest)
            b = self.New_Check(x_new)
            if b == True :
                continue

        time_sampling_end = time.time()
        self.sampling_elapsed += time_sampling_end - time_sampling_start
        
        time_addparent_start = time.time()
        x_new = self.Add_Parent(x_new, x_nearest)
        time_addparent_end = time.time()
        self.addparent_elapsed += (time_addparent_end - time_addparent_start)
        self.nodes.append(x_new)

        time_rewire_start = time.time()
        self.Rewire(x_new)
        time_rewire_end = time.time()
        self.rewire_elapsed += (time_rewire_end - time_rewire_start)
        

        # in goal region
        if self.ingoal_region(x_new):
            self.X_soln.append(x_new)

        self.Graph_sample_num += 1
        if self.Graph_sample_num%100 == 0:
            Graph_cost = self.Cost_Graph()
            self.Graph_data = np.append(self.Graph_data,np.array([[self.Graph_sample_num, Graph_cost]]), axis = 0)

        
        self.e = time.time()

    def print_time(self):
        print("Total time : ", self.e - self.s,"second")
        print("Sampling time : ", self.sampling_elapsed,"second", (self.sampling_elapsed*100)/(self.e-self.s),"%")
        print("Add_Parent time : ", self.addparent_elapsed,"second", (self.addparent_elapsed*100)/(self.e-self.s),"%")
        print("Rewire time : ", self.rewire_elapsed,"second", (self.rewire_elapsed*100)/(self.e-self.s),"%")
        print("Total_iteration = ", self.total_iter)
        print("Cost : ", self.x_goal.cost)


if __name__=="__main__":
    from map.map_loader import grid_map_probability
    from map.taskmap_img_format import map_2d_empty
    from planner.research_rrtstar.rrtstar_probabilty_2d import node, rrt_star
    np.random.seed(1)

 
    # SECTION - Experiment 1
    map_index = 2
    filter_size = 3 # 1 = 3x3, 2 = 5x5, 3 = 7x7
    map = grid_map_probability(map_index, filter_size)
    x_init = np.array([19.5, 110]).reshape(2,1)
    x_goal = np.array([110, 17]).reshape(2,1)


    # SECTION - planner
    distance_weight = 0.5
    obstacle_weight = 0
    rrt = rrt_star(map, x_init, x_goal, distance_weight, obstacle_weight, maxiteration=1000)
    rrt.start_planning()
    path = rrt.Get_Path()


    # SECTION - result
    # rrt.print_time()
    plt.figure(figsize=(10,10))
    plt.axes().set_aspect('equal')
    rrt.Draw_Tree()
    rrt.Draw_path(path)
    plt.imshow(np.transpose(1-map),cmap = "jet", interpolation = 'nearest', origin="lower")
    plt.title("Probability map(same type)")
    plt.show()