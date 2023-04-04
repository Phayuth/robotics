import os
import sys
wd = os.path.abspath(os.getcwd())
sys.path.append(str(wd))

import numpy as np
import matplotlib.pyplot as plt
from collision_check_geometry.collision_class import obj_line2d, obj_point2d, intersect_point_v_rectangle, intersect_line_v_rectangle

class node:
    def __init__(self, x, y, parent=None, cost=0.0) -> None:
        self.x = x
        self.y = y
        self.parent = parent
        self.cost = cost

class rrtstar:
    def __init__(self, map, obstacle_list, startnode, goalnode, eta=None, maxiteration=1000) -> None:
        # map properties
        self.map = map
        self.startnode = node(startnode[0,0], startnode[1,0])
        self.startnode.cost = 0.0
        self.goalnode  = node(goalnode[0,0], goalnode[1,0])
        self.obs = obstacle_list

        # properties of planner
        self.maxiteration = maxiteration
        self.m = self.map.shape[0] * self.map.shape[1]
        self.radius = (2 * (1 + 1/2)**(1/2)) * (self.m/np.pi)**(1/2)
        self.eta = self.radius * (np.log(self.maxiteration) / self.maxiteration)**(1/2)

        # start with a tree vertex have start node and empty branch
        self.tree_vertex = [self.startnode]

    def planning(self):
        for itera in range(self.maxiteration):
            print(itera)
            x_rand = self.sampling()
            x_nearest = self.nearest_node(x_rand)
            x_new = self.steer(x_nearest, x_rand)
            x_new.parent = x_nearest
            x_new.cost = x_new.parent.cost + self.cost_line(x_new, x_new.parent) # add the vertex new, which we have to calculate the cost and add parent as well
            if self.collision_check_node(x_new) or self.collision_check_line(x_new.parent, x_new):
                continue
            else:
                # connect along minimum cost path (choose parent) update the parent of the node of x_new
                X_near = self.near(x_new, self.eta) # find a list of node in the area of x_new
                x_min = x_new.parent # create a xmin variable that hold the minimun node to x_new, which is its parent
                c_min = x_min.cost + self.cost_line(x_min, x_new) # create a variable to hold the current cost of travel from x_start to x_new_parent + x_new_parent to x_new
                for x_near in X_near: # check the node in the area round x_new
                    if self.collision_check_line(x_near, x_new): # check collision from x_new to all node in the area
                        continue
                    # c_min = x_nearest.cost + self.cost_line(x_new, x_nearest)
                    c_new = x_near.cost + self.cost_line(x_near, x_new) # for each node in the area, calculate the cost from new x_start to node in the area + node in the area to x_new
                    if c_new < c_min: # if the new cost is lower than the current cost
                            x_min = x_near # update the node that in the area to be the new x_min
                            c_min = c_new # update the cost of node in the area to be the new c_min
                
                # we update the parent of x_new to be x_min and the x_new cost to be c_new
                x_new.parent = x_min
                x_new.cost = c_min
                self.tree_vertex.append(x_new) # than we commit to the vertex list

                # rewire (update the parent of node surround the area of x_new)
                for x_near in X_near:
                    if self.collision_check_line(x_near, x_new):
                        continue
                    c_near = x_near.cost #  c_near store the cost of from x_start to x_near (near the x_new)
                    c_new = x_new.cost + self.cost_line(x_new, x_near) # find the cost from x_new to each of node near it area
                    if c_new < c_near: 
                        x_near.parent = x_new # update the parent of the x_near to be x_new
                        x_near.cost = x_new.cost + self.cost_line(x_new, x_near) # update the cost of x_near to be cost from start to x_new + the cost from x_new to x_near

    def search_path(self):
        X_near = self.near(self.goalnode, self.eta)
        for x_near in X_near:
            if self.collision_check_line(x_near, self.goalnode):
                continue
            self.goalnode.parent = x_near

            path = [self.goalnode]
            curr_node = self.goalnode

            while curr_node != self.startnode:
                curr_node = curr_node.parent
                path.append(curr_node)

            path.reverse()

            best_path = path

            cost = sum(i.cost for i in path)

            if cost < sum(j.cost for j in best_path):
                best_path = path
        
        return best_path

    def sampling(self):
        x = np.random.uniform(low=0, high=self.map.shape[0])
        y = np.random.uniform(low=0, high=self.map.shape[1])
        x_rand = node(x, y)
        return x_rand
    
    def nearest_node(self, x_rand):
        vertex_list = []
        for each_vertex in self.tree_vertex:
            dist_x = x_rand.x - each_vertex.x
            dist_y = x_rand.y - each_vertex.y
            dist = np.linalg.norm([dist_x, dist_y])
            vertex_list.append(dist)
        min_index = np.argmin(vertex_list)
        x_near = self.tree_vertex[min_index]
        return x_near

    def steer(self, x_nearest, x_rand):
        dist_x = x_rand.x - x_nearest.x
        dist_y = x_rand.y - x_nearest.y
        dist = np.linalg.norm([dist_x, dist_y])
        if dist <= self.eta:
            x_new = x_rand
        else:
            direction = np.arctan2(dist_y, dist_x)
            new_x = self.eta*np.cos(direction) + x_nearest.x
            new_y = self.eta*np.sin(direction) + x_nearest.y
            x_new = node(new_x, new_y)

        return x_new

    def near(self, x_new, min_step):
        neighbor = []
        for index, vertex in enumerate(self.tree_vertex):
            dist = np.linalg.norm([(x_new.x - vertex.x), (x_new.y - vertex.y)])
            if dist <= min_step:
                neighbor.append(index)
        return [self.tree_vertex[i] for i in neighbor]

    def cost_line(self, xstart, xend):
        return np.linalg.norm([(xstart.x - xend.x), (xstart.y - xend.y)]) # simple euclidean distance as cost

    def collision_check_node(self, x_new):
        nodepoint = obj_point2d(x_new.x, x_new.y)

        col = []
        for obs in self.obs:
            colide = intersect_point_v_rectangle(nodepoint, obs)
            col.append(colide)

        if True in col:
            return True
        else:
            return False

    def collision_check_line(self, x_nearest, x_new):
        line = obj_line2d(x_nearest.x, x_nearest.y, x_new.x, x_new.y)

        col = []
        for obs in self.obs:
            colide = intersect_line_v_rectangle(line, obs)
            col.append(colide)

        if True in col:
            return True
        else:
            return False

    def plot_env(self):
        # plot obstacle
        for obs in self.obs:
            obs.plot()

        # plot tree vertex and start and goal node
        for j in self.tree_vertex:
            plt.scatter(j.x, j.y, color="red")

        # plot tree branh
        for k in self.tree_vertex:
            if k is not self.startnode:
                plt.plot([k.x, k.parent.x], [k.y, k.parent.y], color="green")

        # plot start and goal node
        plt.scatter([self.startnode.x, self.goalnode.x], [self.startnode.y, self.goalnode.y], color='cyan')

        # goal node circle
        theta = np.linspace(0,2*np.pi,90)
        x = self.eta * np.cos(theta) + self.goalnode.x
        y = self.eta * np.sin(theta) + self.goalnode.y
        plt.plot(x,y)

if __name__ == "__main__":
    from map.taskmap_geo_format import task_rectangle_obs_7
    from map.taskmap_img_format import bmap
    from map.map_format_converter import img_to_geo
    np.random.seed(9)


    # SECTION - Experiment 1
    # start = np.array([4,4]).reshape(2,1)
    # goal = np.array([7,8]).reshape(2,1)
    # map = np.ones((10,10))
    # obslist = task_rectangle_obs_7()


    # SECTION - Experiment 2
    start = np.array([4,4]).reshape(2,1)
    goal = np.array([8.5,1]).reshape(2,1)
    map = np.ones((10,10))
    obslist = img_to_geo(bmap(), minmax=[0,10], free_space_value=1)


    # SECTION - plot task space
    plt.scatter([start[0,0], goal[0,0]], [start[1,0], goal[1,0]])
    for o in obslist:
        o.plot()
    plt.show()


    # SECTION - Planing Section
    planner = rrtstar(map, obslist, start, goal, maxiteration=1000)
    planner.planning()
    path = planner.search_path()


    # SECTION - plot planning result
    planner.plot_env()
    plt.plot([node.x for node in path], [node.y for node in path], color='blue')
    plt.show()
