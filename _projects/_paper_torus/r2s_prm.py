import networkx as nx
import numpy as np
import os


class SequentialPRM:

    def __init__(self):
        self.rsrc = os.environ["RSRC_DIR"]
        self.graph = nx.read_graphml(
            os.path.join(self.rsrc, "rnd_torus", "saved_planner.graphml")
        )

        # this not actually euclidean data, it is adjacency matrix
        self.graphmatrix = nx.to_numpy_array(self.graph)

        self.coords = self.euclidean_matrix()

    def euclidean_matrix(self):
        coords = np.array(
            [
                list(map(float, self.graph.nodes[n]["coords"].split(",")))
                for n in self.graph.nodes
            ]
        )
        return coords

    def nearest_node(self, query):
        dist = np.linalg.norm(self.coords - query, axis=1)
        min_idx = np.argmin(dist)
        return min_idx, f"n{min_idx}", self.graph.nodes[f"n{min_idx}"]

    def get_xy(self, node):
        coords = self.graph.nodes[node]["coords"].split(",")
        return float(coords[0]), float(coords[1])

    def query_path(self, start_np, end_np):
        _, start, _ = self.nearest_node(start_np)
        _, end, _ = self.nearest_node(end_np)

        path = nx.shortest_path(self.graph, source=start, target=end)
        path_coords = [self.get_xy(node) for node in path]
        path_coords.insert(0, (start_np[0], start_np[1]))
        path_coords.append((end_np[0], end_np[1]))

        return np.array(path_coords)


if __name__ == "__main__":
    prm = SequentialPRM()

    start = np.array([0.0, 0.0])
    end = np.array([-4, -6])
    path = prm.query_path(start, end)
    print(f"Path from {start} to {end} is {path}")
