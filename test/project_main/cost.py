from geometry import dist_between_points
from geometry import steer
from skimage.measure import profile_line
import numpy as np


def Obstacle_cost(x_near, x_new, map):

    line = profile_line(map, x_near, x_new, linewidth=1, mode='constant')
    num = len(line)

    cost = 1 - line

    obs_cost = np.sum(cost) / num

    return obs_cost


def cost_to_go(map, a: tuple, b: tuple) -> float:
    """
    :param a: current location
    :param b: next location
    :return: estimated segment_cost-to-go from a to b
    """

    return (0.8 * (dist_between_points(a, b) / 4)) + (0.2 * Obstacle_cost(a, b, map))


def path_cost(E, a, b, map):
    """
    Cost of the unique path from x_init to x
    :param E: edges, in form of E[child] = parent
    :param a: initial location
    :param b: goal location
    :return: segment_cost of unique path from x_init to x
    """

    cost = 0
    while not b == a:
        p = E[b]
        cost += (0.8 * (dist_between_points(b, p) / 4)) + (0.2 * Obstacle_cost(b, p, map))
        b = p

    return cost


def segment_cost(a, b, map):
    """
    Cost function of the line between x_near and x_new
    :param a: start of line
    :param b: end of line
    :return: segment_cost function between a and b
    """

    return (0.8 * dist_between_points(a, b) / 4) + (0.2 * Obstacle_cost(a, b, map))


def sampling(map):

    row = map.shape[0]

    p = np.ravel(map) / np.sum(map)

    x_sample = np.random.choice(len(p), p=p)

    x = x_sample // row
    y = x_sample % row

    x_rand = np.array([x, y])

    return x_rand


def new_and_near(self, tree, q, map):
    """
    Return a new steered vertex and the vertex in tree that is nearest
    :param tree: int, tree being searched
    :param q: length of edge when steering
    :return: vertex, new steered vertex, vertex, nearest vertex in tree to new vertex
    """
    # x_rand = self.X.sample_free()
    # x_nearest = self.get_nearest(tree, x_rand)
    # x_new = self.bound_point(steer(x_nearest, x_rand, q[0]))
    while True:
        x_rand = self.X.sample_free()
        x_nearest = self.get_nearest(tree, x_rand)
        x_new = self.bound_point(steer(x_nearest, x_rand, q[0]))

        x_pob = np.around(x_new)

        if x_pob[0] >= 100:
            x_pob[0] = 99
        if x_pob[1] >= 100:
            x_pob[1] = 99
        x_pob = map[int(x_pob[0]), int(x_pob[1])]
        p = np.random.uniform(0, 1)
        # print(x_pob,p)
        if x_pob > p:
            break

    # print(dist_between_points(x_rand,x_nearest),dist_between_points(x_new,x_nearest))
    # check if new point is in X_free and not already in V
    if not self.trees[0].V.count(x_new) == 0:  # or not self.X.obstacle_free(x_new):
        return None, None
    self.samples_taken += 1
    return x_new, x_nearest
