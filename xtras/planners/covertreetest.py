import math
import numpy as np
np.random.seed(9)


class CoverTree:
    class Node:
        def __init__(self, point, level):
            self.point = point
            self.children = []
            self.level = level

    def __init__(self, base=2):
        self.base = base
        self.root = None
        self.max_level = None

    def dist(self, p1, p2):
        return math.dist(p1, p2)  # Euclidean distance

    def insert(self, point):
        if self.root is None:
            # If the tree is empty, initialize the root
            self.root = self.Node(point, level=0)
            self.max_level = 0
        else:
            self._insert(self.root, point, self.max_level)

    def _insert(self, node, point, level):
        if level < 0:
            # Stop if we reach a level below zero
            return

        # Compute distance between the new point and the current node's point
        dist_to_node = self.dist(node.point, point)

        # Check if the new point can be a child of the current node
        if dist_to_node <= self.base**level:
            for child in node.children:
                if self.dist(child.point, point) <= self.base ** (level - 1):
                    # Recur on the child if it's a better candidate
                    self._insert(child, point, level - 1)
                    return

            # If no child can hold the new point, create a new child
            new_node = self.Node(point, level - 1)
            node.children.append(new_node)
        else:
            # If the point is too far, recurse to a higher level
            self._insert(node, point, level + 1)

    def nearest_neighbor(self, query_point):
        if self.root is None:
            return None
        return self._nearest_neighbor(self.root, query_point, self.max_level, self.root.point, self.dist(self.root.point, query_point))

    def _nearest_neighbor(self, node, query_point, level, best_point, best_dist):
        dist_to_node = self.dist(node.point, query_point)

        if dist_to_node < best_dist:
            best_point, best_dist = node.point, dist_to_node

        # Check the children nodes that might have closer points
        for child in node.children:
            if self.dist(child.point, query_point) <= best_dist + self.base**level:
                best_point, best_dist = self._nearest_neighbor(child, query_point, level - 1, best_point, best_dist)

        return best_point, best_dist


# Create a cover tree
tree = CoverTree()

for i in range(1000000):
    # Insert points
    tree.insert((np.random.uniform(), np.random.uniform(), np.random.uniform(), np.random.uniform(), np.random.uniform(), np.random.uniform()))
    print(i)

# Query nearest neighbor
point, distance = tree.nearest_neighbor((0.5, 0.8, 0.8, 0.8, 0.8, 0.8))
print(f"Nearest neighbor: {point}, Distance: {distance}")
