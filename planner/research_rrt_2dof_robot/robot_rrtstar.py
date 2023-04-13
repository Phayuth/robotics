""" Path Planning for Planar RR with RRT Star
- Map : create from image to geometry with MapLoader and MapClass
- Collision : geometry check
- Searcher : Cost search

"""

import os
import sys

wd = os.path.abspath(os.getcwd())
sys.path.append(str(wd))

import numpy as np
import matplotlib.pyplot as plt
from collision_check_geometry.collision_class import ObjLine2D, ObjPoint2D, intersect_point_v_rectangle, intersect_line_v_rectangle


class node:

    def __init__(self, x, y, parent=None, cost=0.0) -> None:
        self.x = x
        self.y = y
        self.parent = parent
        self.cost = cost


class RobotRRTStar:

    def __init__(self, mapclass, x_start, x_goal, eta=None, maxiteration=1000) -> None:
        # map properties
        self.mapclass = mapclass
        self.x_min = self.mapclass.xmin
        self.x_max = self.mapclass.xmax
        self.y_min = self.mapclass.ymin
        self.y_max = self.mapclass.ymax
        self.x_start = node(x_start[0, 0], x_start[1, 0])
        self.x_start.cost = 0.0
        self.x_goal = node(x_goal[0, 0], x_goal[1, 0])
        self.obs = self.mapclass.costmap2geo()

        # properties of planner
        self.maxiteration = maxiteration
        self.m = (self.x_max - self.x_min) * (self.y_max - self.y_min)
        self.radius = (2 * (1 + 1/2)**(1 / 2)) * (self.m / np.pi)**(1 / 2)
        self.eta = self.radius * (np.log(self.maxiteration) / self.maxiteration)**(1 / 2)
        self.tree_vertex = [self.x_start]

    def planning(self):
        for itera in range(self.maxiteration):
            print(itera)
            x_rand = self.sampling()
            x_nearest = self.nearest_node(x_rand)
            x_new = self.steer(x_nearest, x_rand)
            x_new.parent = x_nearest
            x_new.cost = x_new.parent.cost + self.cost_line(x_new, x_new.parent)
            if self.collision_check_node(x_new) or self.collision_check_line(x_new.parent, x_new):
                continue
            else:
                X_near = self.near(x_new, self.eta)
                x_min = x_new.parent
                c_min = x_min.cost + self.cost_line(x_min, x_new)
                for x_near in X_near:
                    if self.collision_check_line(x_near, x_new):
                        continue

                    c_new = x_near.cost + self.cost_line(x_near, x_new)
                    if c_new < c_min:
                        x_min = x_near
                        c_min = c_new

                x_new.parent = x_min
                x_new.cost = c_min
                self.tree_vertex.append(x_new)

                for x_near in X_near:
                    if self.collision_check_line(x_near, x_new):
                        continue
                    c_near = x_near.cost
                    c_new = x_new.cost + self.cost_line(x_new, x_near)
                    if c_new < c_near:
                        x_near.parent = x_new
                        x_near.cost = x_new.cost + self.cost_line(x_new, x_near)

    def search_path(self):
        X_near = self.near(self.x_goal, self.eta)
        for x_near in X_near:
            if self.collision_check_line(x_near, self.x_goal):
                continue
            self.x_goal.parent = x_near

            path = [self.x_goal]
            curr_node = self.x_goal

            while curr_node != self.x_start:
                curr_node = curr_node.parent
                path.append(curr_node)

            path.reverse()
            best_path = path
            cost = sum(i.cost for i in path)

            if cost < sum(j.cost for j in best_path):
                best_path = path

        return best_path

    def sampling(self):
        x = np.random.uniform(low=self.x_min, high=self.x_max)
        y = np.random.uniform(low=self.y_min, high=self.y_max)
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
            new_x = self.eta * np.cos(direction) + x_nearest.x
            new_y = self.eta * np.sin(direction) + x_nearest.y
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
        return np.linalg.norm([(xstart.x - xend.x), (xstart.y - xend.y)])

    def collision_check_node(self, x_new):
        nodepoint = ObjPoint2D(x_new.x, x_new.y)

        col = []
        for obs in self.obs:
            colide = intersect_point_v_rectangle(nodepoint, obs)
            col.append(colide)

        if True in col:
            return True
        else:
            return False

    def collision_check_line(self, x_nearest, x_new):
        line = ObjLine2D(x_nearest.x, x_nearest.y, x_new.x, x_new.y)

        col = []
        for obs in self.obs:
            colide = intersect_line_v_rectangle(line, obs)
            col.append(colide)

        if True in col:
            return True
        else:
            return False

    def plot_env(self, after_plan=False):
        # plot obstacle
        for obs in self.obs:
            obs.plot()

        if after_plan:
            # plot tree vertex and branches
            for j in self.tree_vertex:
                plt.scatter(j.x, j.y, color="red")  # vertex
                if j is not self.x_start:
                    plt.plot([j.x, j.parent.x], [j.y, j.parent.y], color="green")  # branch

        # plot start and goal node
        plt.scatter([self.x_start.x, self.x_goal.x], [self.x_start.y, self.x_goal.y], color='cyan')


if __name__ == "__main__":
    from map.taskmap_img_format import bmap
    from map.mapclass import MapLoader, MapClass
    np.random.seed(9)


    # SECTION - Experiment 1
    maploader = MapLoader.loadarray(bmap())
    mapclass = MapClass(maploader=maploader, maprange=[[-np.pi, np.pi], [-np.pi, np.pi]])
    start = np.array([0, 0]).reshape(2, 1)
    goal = np.array([1, 1]).reshape(2, 1)


    # SECTION - planning section
    planner = RobotRRTStar(mapclass, start, goal, maxiteration=1000)
    planner.plot_env()
    plt.show()
    planner.planning()
    path = planner.search_path()


    # SECTION - plot result
    planner.plot_env(after_plan=True)
    plt.plot([node.x for node in path], [node.y for node in path], color='blue')
    plt.show()
