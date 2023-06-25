import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def generate_random_node():
    point = np.random.uniform(-1, 1, size=3)
    point /= np.linalg.norm(point)
    return point

def generate_child_nodes(parent, num_children, distance_threshold):
    children = []
    while len(children) < num_children:
        child = generate_random_node()
        distance = np.linalg.norm(child - parent)
        if distance <= distance_threshold:
            children.append(child)
    return children

# Generate parent node
parent = generate_random_node()

# Generate child nodes
num_children = 100
distance_threshold = 0.5
children = generate_child_nodes(parent, num_children, distance_threshold)

# Plot the nodes and lines
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Parent node
ax.scatter(parent[0], parent[1], parent[2], color='red', label='Parent')

# Child nodes
for child in children:
    ax.scatter(child[0], child[1], child[2], color='blue', label='Child')
    # ax.plot([parent[0], child[0]], [parent[1], child[1]], [parent[2], child[2]], color='black')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# ax.legend()
plt.show()
