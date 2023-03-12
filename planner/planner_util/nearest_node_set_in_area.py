import numpy as np
import matplotlib.pyplot as plt

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

if __name__=="__main__":

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