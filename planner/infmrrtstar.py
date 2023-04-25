""" Informed RRT star 2D.
Map Type : Continuous configuration space 2D
Sampling Method : Informed Sampling + Uniform
Collsion : Geometry based
Path Searcher : Cosider all nodes in the radius of x_goal. -> Calculate Cost to x_start -> Choose best with the lowest cost.
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


class InfmRrtstar:

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

        if mapclass.__class__.__name__ == "CostMapClass":
            self.obs = self.mapclass.costmap2geo()
        else:
            self.obs = self.mapclass.obj

        # properties of planner
        self.maxiteration = maxiteration
        self.m = (self.x_max - self.x_min) * (self.y_max - self.y_min)
        self.radius = (2 * (1 + 1/2)**(1 / 2)) * (self.m / np.pi)**(1 / 2)
        self.eta = self.radius * (np.log(self.maxiteration) / self.maxiteration)**(1 / 2)

        # start with a tree vertex have start node and empty branch
        self.tree_vertex = [self.x_start]
        self.X_soln = []

    def planning(self):
        c_best = np.inf
        for itera in range(self.maxiteration):
            print(itera)
            for x_soln in self.X_soln:
                c_best = x_soln.parent.cost + self.cost_line(x_soln.parent, x_soln) + self.cost_line(x_soln, self.x_goal)
                if x_soln.parent.cost + self.cost_line(x_soln.parent, x_soln) + self.cost_line(x_soln, self.x_goal) < c_best:
                    c_best = x_soln.parent.cost + self.cost_line(x_soln.parent, x_soln) + self.cost_line(x_soln, self.x_goal)

            x_rand = self.sampling(self.x_start, self.x_goal, c_best)

            x_nearest = self.nearest_node(x_rand)
            x_new = self.steer(x_nearest, x_rand)
            x_new.parent = x_nearest
            x_new.cost = x_new.parent.cost + self.cost_line(x_new, x_new.parent)  # add the vertex new, which we have to calculate the cost and add parent as well
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

                # in goal region
                if self.ingoal_region(x_new):
                    self.X_soln.append(x_new)

    def search_path(self):
        for xbest in self.X_soln:
            if self.collision_check_line(xbest, self.x_goal):
                continue
            self.x_goal.parent = xbest

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

    def sampling(self, x_start, x_goal, c_max):
        if c_max < np.inf:
            c_min = self.cost_line(x_start, x_goal)
            print(c_max, c_min)
            x_center = np.array([(x_start.x + x_goal.x) / 2, (x_start.y + x_goal.y) / 2]).reshape(2, 1)
            C = self.RotationToWorldFrame(x_start, x_goal)
            r1 = c_max / 2
            r2 = np.sqrt(c_max**2 - c_min**2) / 2
            L = np.diag([r1, r2])
            while True:
                x_ball = self.sampleUnitBall()
                x_rand = (C@L@x_ball) + x_center
                x_rand = node(x_rand[0, 0], x_rand[1, 0])
                if (self.x_min < x_rand.x < self.x_max) and (self.y_min < x_rand.y < self.y_max):  # check if outside configspace
                    break
        else:
            x_rand = self.unisampling()
        return x_rand

    def sampleUnitBall(self):
        r = np.random.uniform(low=0, high=1)
        theta = np.random.uniform(low=0, high=2 * np.pi)
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        return np.array([[x], [y]]).reshape(2, 1)

    def unisampling(self):
        x = np.random.uniform(low=self.x_min, high=self.x_max)
        y = np.random.uniform(low=self.y_min, high=self.y_max)
        x_rand = node(x, y)
        return x_rand

    def ingoal_region(self, x_new):
        if np.linalg.norm([self.x_goal.x - x_new.x, self.x_goal.y - x_new.y]) <= 1:  # self.eta:
            return True
        else:
            return False

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
        return np.linalg.norm([(xstart.x - xend.x), (xstart.y - xend.y)])  # simple euclidean distance as cost

    def RotationToWorldFrame(self, x_start, x_goal):
        theta = np.arctan2((x_goal.y - x_start.y), (x_goal.x - x_start.x))

        R = np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]]).T

        return R

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

        # plot ingoal region node
        for l in self.X_soln:
            plt.scatter(l.x, l.y, color="yellow")


if __name__ == "__main__":
    from map.taskmap_geo_format import task_rectangle_obs_7
    from map.taskmap_img_format import bmap
    from map.mapclass import CostMapLoader, CostMapClass, GeoMapClass
    np.random.seed(9)

    # SECTION - Experiment 1
    start = np.array([4, 4]).reshape(2, 1)
    goal = np.array([7, 8]).reshape(2, 1)
    mapclass = GeoMapClass(geomap=task_rectangle_obs_7(), maprange=[[0, 10], [0, 10]])

    # SECTION - Experiment 2
    # start = np.array([4, 4]).reshape(2, 1)
    # goal = np.array([8.5, 1]).reshape(2, 1)
    # maploader = CostMapLoader.loadarray(bmap())
    # mapclass = CostMapClass(maploader, maprange=[[0, 10], [0, 10]])

    # SECTION - Planing Section
    planner = InfmRrtstar(mapclass=mapclass, x_start=start, x_goal=goal, maxiteration=500)
    planner.plot_env()
    plt.show()
    planner.planning()
    path = planner.search_path()

    # SECTION - plot planning result
    planner.plot_env(after_plan=True)
    plt.plot([node.x for node in path], [node.y for node in path], color='blue')
    plt.show()
