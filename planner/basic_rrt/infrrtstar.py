import os
import sys
wd = os.path.abspath(os.getcwd())
sys.path.append(str(wd))

import numpy as np
import matplotlib.pyplot as plt
from collision_check_geometry.collision_class import line_obj, point2d_obj, intersect_point_v_rectangle, intersect_line_v_rectangle
from map.taskmap_geo_format import task_rectangle_obs_7

class node:
    def __init__(self, x, y, parent=None, cost=0.0) -> None:
        self.x = x
        self.y = y
        self.parent = parent
        self.cost = cost

class rrtstar:
    def __init__(self) -> None:
        # properties of planner
        self.maxiteration = 1000
        self.startnode = node(4, 4)
        self.startnode.cost = 0.0
        self.goalnode = node(7, 8)

        # map
        self.map = np.ones((10, 10))
        self.obs = task_rectangle_obs_7()

        # distance step update per iteration
        m = self.map.shape[0] * self.map.shape[1]
        self.radius = (2 * (1 + 1/2)**(1/2)) * (m/np.pi)**(1/2)
        self.eta = self.radius * (np.log(self.maxiteration) / self.maxiteration)**(1/2)

        # start with a tree vertex have start node and empty branch
        self.tree_vertex = [self.startnode]
        self.X_soln = []

    def planning(self):
        c_best = np.inf
        for _ in range(self.maxiteration):
            for x_soln in self.X_soln:
                c_best = x_soln.cost
            #     if x_soln.cost < c_best:
            #         c_best = x_soln.cost
            # x_rand = self.sampling(self.startnode, self.goalnode, c_best)

            x_rand = self.unisampling()
            x_nearest = self.nearest_node(x_rand)
            x_new = self.steer(x_nearest, x_rand)
            x_new.parent = x_nearest
            x_new.cost = x_new.parent.cost + self.cost_line(x_new, x_new.parent) # add the vertex new, which we have to calculate the cost and add parent as well
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

    def sampling(self, x_start, x_goal, c_max):
        if c_max < np.inf:
            c_min = np.linalg.norm([x_goal.x - x_start.x], [x_goal.y - x_start.y]).item()
            x_center = np.array([(x_start.x + x_goal.x)/2, (x_start.y + x_goal.y)/2]).reshape(2,1)
            C = self.RotationToWorldFrame(x_start, x_goal)
            r1 = c_max/2
            r2 = np.sqrt(c_max**2 - c_min**2)/2
            L = np.diag([r1, r2])
            x_ball = self.sampleUnitBall()
            x_rand = (C @ L @ x_ball) + x_center
            x_rand = node(x_rand[0,0], x_rand[1,0])
        else:
            x_rand = self.unisampling()
        return x_rand

    def sampleUnitBall(self):
        x, y = np.random.uniform(-1, 1), np.random.uniform(-1, 1)
        if x ** 2 + y ** 2 < 1:
            return np.array([[x], [y]]).reshape(2,1)
        
    def unisampling(self):
        x = np.random.uniform(low=0, high=self.map.shape[0])
        y = np.random.uniform(low=0, high=self.map.shape[1])
        x_rand = node(x, y)
        return x_rand
    
    def ingoal_region(self, x_new):
        if np.linalg.norm([self.goalnode.x - x_new.x, self.goalnode.y - x_new.y]) <= 1 :# self.eta:
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


    def RotationToWorldFrame(self, x_start, x_goal):
        theta = np.arctan2((x_goal.y - x_start.y),(x_goal.x - x_start.x))

        R = np.array([[ np.cos(theta), np.sin(theta)],
                      [-np.sin(theta), np.cos(theta)]])
        
        return R
    
    def collision_check_node(self, x_new):
        # nodepoint = point2d_obj(x_new.x, x_new.y)

        # col = []
        # for obs in self.obs:
        #     colide = intersect_point_v_rectangle(nodepoint, obs)
        #     col.append(colide)

        # if True in col:
        #     return True
        # else:
        #     return False
        return False

    def collision_check_line(self, x_nearest, x_new):
        # line = line_obj(x_nearest.x, x_nearest.y, x_new.x, x_new.y)
        # col = []
        # for obs in self.obs:
        #     colide = intersect_line_v_rectangle(line, obs)
        #     col.append(colide)
        # if True in col:
        #     return True
        # else:
        #     return False
        return False
    
    def plot_env(self):
        # plot obstacle
        # for obs in self.obs:
        #     obs.plot()

        # plot tree vertex and start and goal node
        for j in self.tree_vertex:
            plt.scatter(j.x, j.y, color="red")

        # plot tree branh
        for k in self.tree_vertex:
            if k is not self.startnode:
                plt.plot([k.x, k.parent.x], [k.y, k.parent.y], color="green")

        # plot start and goal node
        plt.scatter([self.startnode.x, self.goalnode.x], [self.startnode.y, self.goalnode.y], color='cyan')

        # plot ingoal region node
        for l in self.X_soln:
            plt.scatter(l.x, l.y, color="yellow")

if __name__ == "__main__":
    np.random.seed(9)
    planner = rrtstar()
    planner.planning()
    planner.plot_env()
    for ind, val in enumerate(planner.X_soln):
        print(val.cost)
    print(len(planner.X_soln))
    # path = planner.search_path()
    # plt.plot([node.x for node in path], [node.y for node in path], color='blue')

    plt.show()