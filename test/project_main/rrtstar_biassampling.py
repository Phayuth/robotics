import os
import sys
wd = os.path.abspath(os.getcwd())
sys.path.append(str(wd))

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
    def __init__(self, map:np.ndarray, x_init:node, x_goal:node, obs:list, obstacle_center:list, collision_range:float, eta:float=None, maxiteration:int=1000):
        """General RRT star method of planning with Bias sampling

        Args:
            map (np.ndarray): occupancy grid map
            x_init (node): start configuration node
            x_goal (node): end configuration node
            eta (float): RRT* constants
            obs (list): list of obstacle generate from [generate obstcle from map]
            obstacle_center (list): list of obstacle center generate from [generate obstcle from map]
            collision_range (float): range of obstacle generate from [generate obstcle from map]
            max_iteration (int): maximum of iteration
        """
        # map properties
        self.map = map
        self.x_init = node(x_init[0,0], x_init[1,0])
        self.x_goal = node(x_goal[0,0], x_goal[1,0])
        self.nodes = [self.x_init]
        self.obs = obs
        self.obstacle_center = obstacle_center
        self.collision_range = collision_range

        # planner properties
        self.maxiteration = maxiteration
        self.m = map.shape[0] * map.shape[1]
        self.r = (2 * (1 + 1/2)**(1/2)) * (self.m/np.pi)**(1/2)
        self.eta = self.r * (np.log(self.maxiteration) / self.maxiteration)**(1/2)

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

    def Bias_sampling(self)-> node:
        """bias sampling method

        Returns:
            node: random bias sampling node
        """
        row = self.map.shape[1]

        p = np.ravel(self.map) / np.sum(self.map)

        x_sample = np.random.choice(len(p), p=p)

        x = x_sample // row
        y = x_sample % row

        x = np.random.uniform(low=x - 0.5, high=x + 0.5)
        y = np.random.uniform(low=y - 0.5, high=y + 0.5)

        x_rand = node(x, y)

        return x_rand

    def Node_collision_check(self, sample:node)-> bool:
        """geometry collision detection. check if the input node is in collision with obstacle

        Args:
            sample (node): input node sample

        Returns:
            bool: return True if it collide with obstacle, False if not
        """
        p = Point(sample.x, sample.y)

        for o in self.obs:
            if p.within(o):
                return False
        return True

    def Edge_collision_check(self, x_nearest:node, x_new:node)-> bool:
        """geometry collision detection. check if a line between 2 nodes x_nearest and x_new is in collision with obstacle or not.
        check with obstacle in obstacle center list
        Args:
            x_nearest (node): nearest node in tree to x rand
            x_new (node): new node in tree created in directoin of x rand

        Returns:
            bool: return True if a segment of line is collide, False if not
        """
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

            if r <= self.collision_range:
                return False
        return True

    def Distance_Cost(self, start:node, end:node)-> float:
        """calculate distance from one node to other node based on euclidean distance norm
        [note] = yeongmin name this incorrectly. to get distance cost, it must divide by eta which done in function line cost
        
        Args:
            start (node): start pose with node class
            end (node): end pose with node class

        Returns:
            float: norm value between 2 point
        """
        distance_cost = np.linalg.norm(start.arr - end.arr)

        return distance_cost

    def Line_Cost(self, start:node, end:node)-> float:
        """calculate cost for node based on distance euclidean norm

        Args:
            start (node): start pse with node class
            end (node): end pose with node class

        Returns:
            float: node cost
        """

        cost = self.Distance_Cost(start, end)/(self.eta)

        return cost

    def Nearest(self, x_rand:node)-> node:
        """Find the nearest node in the tree that is the nearest to the random node

        Args:
            x_rand (node): a random node from bias sampling

        Returns:
            node: the nearest node in tree nearest to the random node x_rand
        """
        vertex = []
        i = 0
        for x_near in self.nodes:

            dist = self.Distance_Cost(x_near, x_rand)
            vertex.append([dist, i, x_near])
            i+=1

        vertex.sort()
        x_nearest = vertex[0][2]

        return x_nearest

    def Steer(self, x_rand:node, x_nearest:node)-> node:
        """steer function to create a new node that is in the direction of x_rand node and near to x_nearest node

        Args:
            x_rand (node): a random node from bias sampling
            x_nearest (node): the nearest node that already in the tree to the random node x_rand

        Returns:
            node: a new node that is near the x_nearest in the direction of x_rand
        """

        d = self.Distance_Cost(x_rand, x_nearest)

        if d < self.eta :
            x_new = node(x_rand.x, x_rand.y)
        else:
            new_x = x_nearest.x + self.eta * ((x_rand.x - x_nearest.x)/d)
            new_y = x_nearest.y + self.eta * ((x_rand.y - x_nearest.y)/d)

            x_new  = node(new_x, new_y)

        return x_new

    def Exist_Check(self, x_new:node)-> bool:
        """check if the new node is already exist in the tree

        Args:
            x_new (node): new created node from Steer function

        Returns:
            bool: True if already in the tree, False if not
        """
        for x_near in self.nodes:
            if x_new.x == x_near.x and x_new.y == x_near.y :
                return False
            else :
                return True

    def Add_Parent(self, x_new:node, x_nearest:node)-> tuple[node,node]:
        """function for adding parent node to x_new node

        Args:
            x_new (node): new created node from Steer function
            x_nearest (node): nearest node in the tree nearest to the x_new

        Returns:
            tuple[node,node]: x_new node that has x_nearest as its parent and cost, x_nearest
        """

        x_new.cost = x_nearest.cost + self.Line_Cost(x_nearest, x_new)
        x_new.parent = x_nearest
        for x_near in self.nodes:
            if self.Distance_Cost(x_near, x_new) <= self.eta and self.Edge_collision_check(x_near, x_new) :
                if  x_near.cost + self.Line_Cost(x_near, x_new) < x_nearest.cost + self.Line_Cost(x_nearest, x_new):

                    x_nearest = x_near
                    x_new.cost = x_nearest.cost + self.Line_Cost(x_nearest, x_new)
                    x_new.parent = x_nearest

        return x_new, x_nearest

    def Rewire(self, x_new:node):
        """reconstruct rrt tree

        Args:
            x_new (node): new node
        """

        for x_near in self.nodes:
            if x_near is not x_new.parent:
                if self.Distance_Cost(x_near, x_new) <= self.eta and self.Edge_collision_check(x_new, x_near) :
                    if x_new.cost + self.Line_Cost(x_new, x_near) < x_near.cost:
                        x_near.parent = x_new
                        x_near.cost = x_new.cost + self.Line_Cost(x_new, x_near)

    def Get_Path(self)-> list:
        """get a list of path from start to end if path is found, else None

        Returns:
            list: list of path
        """

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
        """draw rrt tree on plot
        """
        for i in self.nodes:
            if i is not self.x_init:
                plt.plot([i.x, i.parent.x], [i.y, i.parent.y], "blue")

    def Draw_obs(self):
        """draw obstacle on plot
        """

        fig, axs = plt.subplots(figsize=(10, 10))
        for o in self.obs:

            x, y = o.exterior.xy
            axs.fill(x, y, fc="black", ec="none")

    def Draw_path(self, path:list):
        """draw result path on plot

        Args:
            path (list): list of node obtain from result Get Path function
        """

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
        """start planning loop
        """
        while True:
            # start record sampling time
            time_sampling_start = time.time()
            while True:
                # create random node by starting to sample bias
                x_rand = self.Bias_sampling()
                # update number of iteration
                self.total_iter += 1
                # find nearest node from sampling 
                x_nearest = self.Nearest(x_rand)
                # create a new node in the direction of random and nearest to nearest node
                x_new = self.Steer(x_rand, x_nearest)
                # check whether to add new node to tree or not
                if self.Node_collision_check(x_new):
                    break
                # stop if the sampling iteration reach maximum
                if self.total_iter == self.maxiteration:
                    break
            # stop record sampling time
            time_sampling_end = time.time()
            # determine sampling time elapsed
            self.sampling_elapsed += time_sampling_end - time_sampling_start
            # stop the entire planner if iteraton reach maximum
            if self.total_iter == self.maxiteration:
                break
            # update sample taken number
            self.sample_taken += 1
            print("==>> self.sample_taken: ", self.sample_taken)

            # start record addparent time
            time_addparent_start = time.time()
            # check if line segment between x_nearest and x_new is in collision with obstacle or not
            if self.Edge_collision_check(x_nearest, x_new) == False:
                continue
            # create xnew node with xnearest as its parent
            x_new, x_nearest = self.Add_Parent(x_new, x_nearest)
            # stop record addparent time
            time_addparent_end = time.time()
            # determine addparent time elapsed
            self.addparent_elapsed += time_addparent_end - time_addparent_start
            # add xnew to rrt tree nodes
            self.nodes.append(x_new)

            # start record rewire time
            time_rewire_start = time.time()
            # start rewire tree
            self.Rewire(x_new)
            time_rewire_end = time.time()
            # determine rewire time elapsed
            self.rewire_elapsed += time_rewire_end - time_rewire_start

            # record graph cost data
            self.Graph_sample_num += 1
            if self.Graph_sample_num%100 == 0:
                Graph_cost = self.Cost_Graph()
                self.Graph_data = np.append(self.Graph_data,np.array([[self.Graph_sample_num, Graph_cost]]), axis = 0)
        
        # record end of planner loop
        self.e = time.time()

    def print_time(self):
        print("Total time : ", self.e - self.s,"second")
        print("Sampling time : ", self.sampling_elapsed,"second", (self.sampling_elapsed*100)/(self.e-self.s),"%")
        print("Add_Parent time : ", self.addparent_elapsed,"second", (self.addparent_elapsed*100)/(self.e-self.s),"%")
        print("Rewire time : ", self.rewire_elapsed,"second", (self.rewire_elapsed*100)/(self.e-self.s),"%")
        print("Total_iteration = ", self.total_iter)
        print("Cost : ", self.x_goal.cost)


if __name__=="__main__":
    from map.generate_obstacle_space import Obstacle_generater, obstacle_generate_from_map
    from map.mapclass import CostMapLoader
    np.random.seed(0)


    # SECTION - Experiment 1
    index = 0
    map, obstacle, obstacle_center  = obstacle_generate_from_map(index)
    plt.imshow(map)
    plt.show()
    obs = Obstacle_generater(obstacle)
    collision_range = (2**(1/2))/2
    x_init = np.array([15, 15]).reshape(2,1)
    x_goal = np.array([20, 20]).reshape(2,1)


    # SECTION - planner
    rrt = rrt_star(map, x_init, x_goal, obs, obstacle_center, collision_range, maxiteration=1000)
    rrt.start_planning()
    path = rrt.Get_Path()


    # SECTION - result
    rrt.print_time()
    rrt.Draw_obs()
    rrt.Draw_Tree()
    rrt.Draw_path(path)
    plt.show()