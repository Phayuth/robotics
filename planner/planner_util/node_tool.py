import numpy as np
import random
import matplotlib.pyplot as plt

def sampling_point_step(x_base,y_base,dist_step):
    
    x_sample = random.randint(-10, 10)
    y_sample = random.randint(-10, 10)

    theta = np.arctan2((y_sample - y_base),(x_sample - x_base))
    x_new = dist_step*np.cos(theta)
    y_new = dist_step*np.sin(theta)
    
    return x_new, y_new, x_sample, y_sample

def nearest_neighbor_circle(r, distribution, node):
    
    neighbor = []

    for i in range(len(distribution)):
        dx = distribution[i,0]
        dy = distribution[i,1]

        base = [node[0,0],node[1,0]]

        dist = [(base[0] - dx) - (base[1] - dy)]
        
        if np.linalg.norm(dist) <= r:
            neighbor.append([dx,dy])

    return neighbor

def nearest_node(node_list,node):
    D = []
    for i in range(len(node_list)):
        xi = node_list[i][0]
        yi = node_list[i][1]

        x0 = node[0]
        y0 = node[1]

        coord = [(xi - x0),(yi - y0)]
        d = np.linalg.norm(coord)
        D.append(d)

    nearest_index = np.argmin(D)
    return node_list[nearest_index]

if __name__ == "__main__":
    # sampling point step
    x_base = 0
    y_base = 0
    x_new, y_new, x_sample, y_sample = sampling_point_step(x_base,y_base,dist_step=2)

    print(x_new, y_new)
    print(x_sample, y_sample)
    plt.scatter(0, 0)
    plt.scatter(x_new, y_new)
    plt.scatter(x_sample, y_sample)
    plt.show()

    # nearest neighbor
    r = 2
    list_node = [[0,0],[1,0],[-1,0],[0,1],[0,-1],[0,5]]
    node = np.array([[0],[0]])
    distribution = np.array(list_node)

    neg = nearest_neighbor_circle(r,distribution,node)

    x , y = distribution[:,0] , distribution[:,1]
    x_base, y_base = node[0], node[1]

    print(neg)
    plt.scatter(x,y)
    plt.scatter(x_base,y_base,c="red")
    plt.show()

    # nearest node
    list_node = [[0,0],[0,1],[0,2],[0,3],[0,4],[0,5]]
    node = [1,3]

    list_node_array = np.array(list_node)

    near = nearest_node(list_node,node)

    x , y = list_node_array[:,0] , list_node_array[:,1]
    x_node, y_node = node[0], node[1]

    print(near)

    plt.scatter(x,y)
    plt.scatter(x_node,y_node)
    plt.show()