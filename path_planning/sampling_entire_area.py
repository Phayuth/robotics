import numpy as np
import matplotlib.pyplot as plt

def get_neighbor_in_circle(r,distribution,node):
    
    neighbor = []

    for i in range(len(distribution)):
        dx = distribution[i,0]
        dy = distribution[i,1]

        base = [node[0,0],node[1,0]]

        dist = [(base[0] - dx) - (base[1] - dy)]
        
        if np.linalg.norm(dist) <= r:
            neighbor.append([dx,dy])

    return neighbor

r = 2


list_node = [[0,0],[1,0],[-1,0],[0,1],[0,-1],[0,5]]
node = np.array([[0],[0]])
distribution = np.array(list_node)


# x = np.random.randint(100, size=100)
# x = np.expand_dims(x, axis=1)

# y = np.random.randint(100, size=100)
# y = np.expand_dims(y, axis=1)

# distribution = np.transpose(np.append(x,y,axis=1))


neg = get_neighbor_in_circle(r,distribution,node)

x , y = distribution[:,0] , distribution[:,1]
x_node, y_node = node[0], node[1]

print(neg)

plt.scatter(x,y)
plt.scatter(x_node,y_node)
plt.show()
