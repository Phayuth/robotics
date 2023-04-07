import os
import sys
wd = os.path.abspath(os.getcwd())
sys.path.append(str(wd))

import numpy as np
import matplotlib.pyplot as plt
from collision_check_geometry.collision_class import obj_line2d, obj_point2d, intersect_point_v_rectangle, intersect_line_v_rectangle
from map.map_value_range import map_val, map_multi_val

class node:
    def __init__(self, x, y, parent=None)-> None:
        self.x = x
        self.y = y
        self.parent = parent

class RobotRRT():
    def __init__(self, map, obstacle_list, startnode, goalnode, eta=0.3, maxiteration=1000)-> None:
        # map properties
        self.map = map
        self.startnode = node(startnode[0,0], startnode[1,0])
        self.goalnode  = node(goalnode[0,0], goalnode[1,0])
        self.obs = obstacle_list

        # properties of planner
        self.maxiteration = maxiteration
        self.eta = eta 
        self.tree_vertex = [self.startnode]

    def planing(self):
        for itera in range(self.maxiteration):
            print(itera)
            x_rand = self.sampling()
            x_nearest = self.nearest_node(x_rand)
            x_new = self.steer(x_nearest, x_rand)
            x_new.parent = x_nearest
            if self.collision_check_node(x_new) or self.collision_check_line(x_nearest, x_new):
                continue
            else:
                self.tree_vertex.append(x_new)
    
    def search_path(self):

        x_near_to_goal = self.nearest_node(self.goalnode)
        self.goalnode.parent = x_near_to_goal

        path = [self.goalnode]
        curr_node = self.goalnode

        while curr_node != self.startnode:
            curr_node = curr_node.parent
            path.append(curr_node)

        path.reverse()

        return path

    def sampling(self):
        x = np.random.uniform(low=-np.pi, high=np.pi)
        y = np.random.uniform(low=-np.pi, high=np.pi)
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

        # plot tree branches
        for k in self.tree_vertex:
            if k is not self.startnode:
                plt.plot([k.x, k.parent.x],[k.y, k.parent.y], color="green")

        # plot start and goal node
        plt.scatter([self.startnode.x, self.goalnode.x], [self.startnode.y, self.goalnode.y], color='cyan')

if __name__ == "__main__":
    from map.taskmap_img_format import bmap
    from map.map_format_converter import img_to_geo
    np.random.seed(9)


    # SECTION - Experiment 1
    start = np.array([0,0]).reshape(2,1)
    goal = np.array([1,1]).reshape(2,1)
    map = np.ones((10,10))
    obslist = img_to_geo(bmap(), minmax=[-np.pi,np.pi], free_space_value=1)


    # SECTION - plot task space
    plt.scatter([start[0,0], goal[0,0]], [start[1,0], goal[1,0]])
    for o in obslist:
        o.plot()
    plt.show()


    # SECTION - Planning Section
    planner = RobotRRT(map, obslist, start, goal, eta=0.1, maxiteration=1000)
    planner.planing()
    path = planner.search_path()


    # SECTION - plot planning result
    planner.plot_env()
    plt.plot([node.x for node in path], [node.y for node in path], color='blue')
    plt.show()
