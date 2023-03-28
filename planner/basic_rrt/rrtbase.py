import numpy as np
import matplotlib.pyplot as plt


class node:
    def __init__(self, x, y) -> None:
        self.x = x
        self.y = y


class rrt_base():
    def __init__(self) -> None:
        # properties of planner
        self.maxiteration = 1000
        self.startnode = node(35, 20)
        self.goalnode = node(10, 10)

        # map properties
        self.map = np.ones((50, 50))

        # distance step update per iteration
        self.d_step = 1  # eta

        # start with a tree vertex have start node and empty branch
        self.tree_vertex = [self.startnode]
        self.tree_branch = []

    def planing(self):
        for i in range(self.maxiteration):
            x_rand = self.sampling()
            # print(x_rand.x, x_rand.y)
            # plt.scatter(x_rand.x, x_rand.y, color='black')
            x_nearest = self.nearest_node(x_rand)
            x_new = self.steer(x_nearest, x_rand)
            if self.collision_check_node(x_new):
                break
            if self.collision_check_line(x_nearest, x_new):
                break
            else:
                self.tree_vertex.append(x_new)
                self.tree_branch.append([x_nearest, x_new])

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

        if dist <= self.d_step:
            x_new = x_rand
        else:
            direction = np.arctan2(dist_y, dist_x)
            new_x = self.d_step*np.cos(direction) + x_nearest.x
            new_y = self.d_step*np.sin(direction) + x_nearest.y
            x_new = node(new_x, new_y)
        return x_new

    def collision_check_node(self, x_new):
        return False

    def collision_check_line(self, x_nearest, x_new):
        return False


if __name__ == "__main__":
    np.random.seed(5)
    planner = rrt_base()
    planner.planing()

    for j in planner.tree_vertex:
        plt.scatter(j.x, j.y, color="red")
    for k in planner.tree_branch:
        plt.plot([k[0].x, k[1].x], [k[0].y, k[1].y], color="green")

    plt.scatter([planner.startnode.x, planner.goalnode.x], [planner.startnode.y, planner.goalnode.y], color='cyan')
    plt.show()
