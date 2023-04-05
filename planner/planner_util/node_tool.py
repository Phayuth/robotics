import numpy as np
import matplotlib.pyplot as plt


class node:
    def __init__(self, x, y, cost=0, cost_euld=0, parent=None, child=None):
        self.x = x
        self.y = y
        self.cost = cost
        self.cost_euld = cost_euld
        self.parent = parent
        self.child = child


def steer(q_base, q_rand, dist_step):
    if np.linalg.norm([q_base.x - q_rand.x, q_base.y - q_rand.y]) <= dist_step:
        q_new = q_rand

    else:
        # theta approach
        theta = np.arctan2((q_rand.y - q_base.y), (q_rand.x - q_base.x))
        x_new = dist_step * np.cos(theta) + q_base.x
        y_new = dist_step * np.sin(theta) + q_base.y
        q_new = node(x_new, y_new)

        # hypothenus approach cos=near/hypo, sin=far/hypos
        # hypo = np.linalg.norm([q_base.x - q_rand.x, q_base.y - q_rand.y])
        # x_new = dist_step*(q_rand.x - q_base.x)/hypo + q_base.x
        # y_new = dist_step*(q_rand.y - q_base.y)/hypo + q_base.y
        # q_new = node(x_new, y_new)

    return q_new


def nearest_neighbor(r, nodeslist, node):
    neighbor = []
    for ind, nd in enumerate(nodeslist):
        dist = np.linalg.norm([nd.x - node.x, nd.y - node.y])
        if np.linalg.norm(dist) <= r:
            neighbor.append(nd)

    return neighbor


def nearest_node(nodeslist, node):
    D = []
    for ind, nd in enumerate(nodeslist):
        dist = np.linalg.norm([nd.x - node.x, nd.y - node.y])
        D.append(dist)
    nearest_index = np.argmin(D)

    return nodeslist[nearest_index]


if __name__ == "__main__":

    # SECTION - steer
    q_base = node(0, 0)
    q_rand = node(np.random.randint(5), np.random.randint(5))
    q_new = steer(q_base, q_rand, dist_step=1)
    plt.scatter([q_base.x, q_rand.x, q_new.x], [q_base.y, q_rand.y, q_new.y])
    plt.show()


    # SECTION - nearest neighbor
    r = 2
    list_node = [
        node(0, 0),
        node(1, 0),
        node(-1, 0),
        node(0, 1),
        node(0, -1),
        node(0, 5),
    ]
    nd = node(0, 0)
    neg = nearest_neighbor(r, list_node, node)
    print(neg)


    # SECTION - nearest node
    list_node = [node(0, 0), node(0, 1), node(0, 2), node(0, 3), node(0, 4), node(0, 5)]
    nd = [1, 3]
    near = nearest_node(list_node, node)
    print(near)
