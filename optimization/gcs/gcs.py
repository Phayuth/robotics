import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Circle
import cvxpy as cp
from scipy.spatial import HalfspaceIntersection
from shapely.geometry import LineString, Point
from shapely.geometry import Polygon as SPolygon
from scipy.optimize import linprog


class GCS:

    def __init__(self):
        # Whether to solve the integer program or the convex relaxation
        self.solve_integer = False

        self.start = np.array([0.5, 0.1])
        self.goal = np.array([3.9, 3.9])

    def point_inside_region(self, point, region):
        return not np.any(region.halfspaces @ np.hstack((point, 1)) > 0)

    def order_vertices(self, points):
        center = np.mean(points, axis=0)
        centered_points = points - center
        thetas = np.arctan2(centered_points[:, 1], centered_points[:, 0])
        idxs = np.argsort(thetas)
        return points[idxs]

    def edge_edge_intersection(self, edge1, edge2):
        l1 = LineString(edge1)
        l2 = LineString(edge2)
        return not l1.intersection(l2).is_empty

    def do_regions_intersect(self, region1, region2):
        points1 = self.order_vertices(region1.intersections)
        points2 = self.order_vertices(region2.intersections)
        for i in range(len(points1)):
            for j in range(len(points2)):
                if self.edge_edge_intersection(points1[[i, (i + 1) % len(points1)]], points2[[j, (j + 1) % len(points2)]]):
                    return True
        return False

    def construct_overlap_adj_mat(self, regions):
        foo = 2 + len(region_tuples)
        adj_mat = np.zeros((foo, foo))
        start_idx = len(regions)
        goal_idx = len(regions) + 1
        for i in range(len(regions)):
            adj_mat[i, start_idx] = adj_mat[start_idx, i] = self.point_inside_region(self.start, regions[i])
            adj_mat[i, goal_idx] = adj_mat[goal_idx, i] = self.point_inside_region(self.goal, regions[i])
            for j in range(i + 1, len(regions)):
                adj_mat[i, j] = adj_mat[j, i] = self.do_regions_intersect(regions[i], regions[j])
        # Remove extra edges between the start/goal point and other containing regions
        # This is a hacky way to avoid weird issues with having multiple "start" or "goal" regions
        # in the conjugate graph
        zero_it_start = False
        zero_it_goal = False
        for i in range(adj_mat.shape[0]):
            if zero_it_start:
                adj_mat[start_idx, i] = adj_mat[i, start_idx] = 0
            if adj_mat[start_idx, i]:
                zero_it_start = True
            if zero_it_goal:
                adj_mat[goal_idx, i] = adj_mat[i, goal_idx] = 0
            if adj_mat[goal_idx, i]:
                zero_it_goal = True
        return adj_mat

    def compute_halfspace_intersection(self, region1, region2):
        # See: https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.HalfspaceIntersection.html
        halfspaces = np.vstack((region1.halfspaces, region2.halfspaces))
        norm_vector = np.reshape(np.linalg.norm(halfspaces[:, :-1], axis=1), (halfspaces.shape[0], 1))
        c = np.zeros((halfspaces.shape[1],))
        c[-1] = -1
        A = np.hstack((halfspaces[:, :-1], norm_vector))
        b = -halfspaces[:, -1:]
        res = linprog(c, A_ub=A, b_ub=b, bounds=(None, None))
        x = res.x[:-1]
        y = res.x[-1]
        new_halfspace = HalfspaceIntersection(halfspaces, x, incremental=False)
        return new_halfspace, x

    def construct_gcs_regions(self, overlap_adj_mat, halfspace_reps):
        gcs_regions_idx_dict = dict()
        gcs_regions = []
        region_points = []
        start_idx = len(overlap_adj_mat) - 2
        goal_idx = len(overlap_adj_mat) - 1
        for i in range(overlap_adj_mat.shape[0]):
            for j in range(i, overlap_adj_mat.shape[1]):
                if overlap_adj_mat[i, j]:
                    gcs_regions_idx_dict[(i, j)] = gcs_regions_idx_dict[(j, i)] = len(gcs_regions)
                    if i == start_idx or j == start_idx:
                        new_region = "start"
                        point = self.start
                    elif i == goal_idx or j == goal_idx:
                        new_region = "goal"
                        point = self.goal
                    else:
                        new_region, point = self.compute_halfspace_intersection(halfspace_reps[i], halfspace_reps[j])
                    gcs_regions.append(new_region)
                    region_points.append(point)
        region_points = np.array(region_points)

        # Construct conjugate graph
        num_regions = len(gcs_regions)
        conjugate_graph = np.zeros((num_regions, num_regions))
        for i in range(overlap_adj_mat.shape[0]):
            for j in range(overlap_adj_mat.shape[1]):
                if overlap_adj_mat[i, j]:
                    for k in range(overlap_adj_mat.shape[0]):
                        if k == i or k == j:
                            continue
                        e1 = gcs_regions_idx_dict[(i, j)]
                        if overlap_adj_mat[i, k]:
                            e2 = gcs_regions_idx_dict[(i, k)]
                            conjugate_graph[e1, e2] = conjugate_graph[e2, e1] = 1
                        if overlap_adj_mat[j, k]:
                            e2 = gcs_regions_idx_dict[(j, k)]
                            conjugate_graph[e1, e2] = conjugate_graph[e2, e1] = 1

        return gcs_regions, region_points, gcs_regions_idx_dict, conjugate_graph

    def solve_gcs_rounding(self, gcs_regions, adj_mat):
        # Set up dictionaries to hold all of the variables in an organized fashion
        y_vars = dict()  # One R^2 variable for each edge (u,v)
        z_vars = dict()  # One R^2 variable for each edge (u,v)
        phi_vars = dict()  # One [0,1] variable for each edge (u,v)
        l_vars = dict()  # One non-negative slack variable for each edge (u,v)

        start_idx = gcs_regions.index("start")
        goal_idx = gcs_regions.index("goal")

        # Create all of the decision variables
        for i in range(len(adj_mat)):
            for j in range(len(adj_mat)):
                if adj_mat[i, j]:
                    y_vars[(i, j)] = cp.Variable(2)
                    z_vars[(i, j)] = cp.Variable(2)
                    if self.solve_integer:
                        phi_vars[(i, j)] = cp.Variable(integer=True)
                    else:
                        phi_vars[(i, j)] = cp.Variable()
                    l_vars[(i, j)] = cp.Variable()

        constraints = []
        objective = 0

        # Construct the objective function
        for edge, l_var in l_vars.items():
            #
            objective += l_var  # The l slack variables are used with the Euclidean perspective function (GCS 1, Page 16)

        # Slack variable constraints
        # (GCS 1, Page 16)
        for l_var in l_vars.values():
            #
            constraints += [l_var >= 0]

        # Second Order Cone Constraints
        # (GCS 1, Page 16, EQ 23) *but using the norm, not the squared norm
        for edge in y_vars.keys():
            y = y_vars[edge]
            z = z_vars[edge]
            phi = phi_vars[edge]
            l = l_vars[edge]
            constraints += [cp.SOC(l, y - z)]  # Note: the extra phi_e doesn't appear on the LHS, because the norm isn't squared
            # I don't remember exactly why, but Mark says so, and I trust Mark

        # Set Membership Constraints
        # (GCS 1, Page 15, EQ 21b)
        for edge in y_vars.keys():
            y = y_vars[edge]
            z = z_vars[edge]
            phi = phi_vars[edge]

            # For all of these cases, use the singleton set special case
            # (GCS 1, Page 17, 6.2.1)
            if edge[0] == start_idx:
                constraints += [y == phi * self.start]
            elif edge[0] == goal_idx:
                constraints += [y == phi * self.goal]
            else:
                # General case (GCS 1, Page 18, 6.2.2)
                A = gcs_regions[edge[0]].halfspaces[:, :-1]
                b = -gcs_regions[edge[0]].halfspaces[:, -1]
                constraints += [A @ y <= b * phi]

            if edge[1] == start_idx:
                constraints += [z == phi * self.start]
            elif edge[1] == goal_idx:
                constraints += [z == phi * self.goal]
            else:
                # General case (GCS 1, Page 18, 6.2.2)
                A = gcs_regions[edge[1]].halfspaces[:, :-1]
                b = -gcs_regions[edge[1]].halfspaces[:, -1]
                constraints += [A @ z <= b * phi]

        # Regular Conservation of Flow
        # (GCS 1, Page 15, EQ 21c)
        v_in_flows = dict()
        v_out_flows = dict()
        # print("Ordinary Conservation of Flow")
        for vertex in range(len(adj_mat)):
            # By convention, row denotes source vertex and column denotes target vertex for directed edges
            in_idxs = np.nonzero(adj_mat[:, vertex])[0]
            out_idxs = np.nonzero(adj_mat[vertex, :])[0]
            in_offset = 1 if vertex == start_idx else 0
            out_offset = 1 if vertex == goal_idx else 0
            in_flow = cp.sum([phi_vars[(in_idx, vertex)] for in_idx in in_idxs]) + in_offset
            out_flow = cp.sum([phi_vars[(vertex, out_idx)] for out_idx in out_idxs]) + out_offset
            constraints += [in_flow == out_flow]
            constraints += [out_flow <= 1]
            # print(str(vertex) + "\tin: " + str(in_idxs) + "\tout: " + str(out_idxs))
            # print(str(vertex) + "\tin: " + str(in_flow) + "\tout: " + str(out_flow))
            v_in_flows[vertex] = in_flow
            v_out_flows[vertex] = out_flow

        # Spatial Conservation of Flow
        # (GCS 1, Page 15, EQ 21d)
        # print("Spatial Conservation of Flow")
        for vertex in range(len(adj_mat)):  # Ignore the start and goal vertices
            if vertex == start_idx or vertex == goal_idx:
                continue
            in_idxs = np.nonzero(adj_mat[:, vertex])[0]
            out_idxs = np.nonzero(adj_mat[vertex, :])[0]
            in_flow = cp.sum([z_vars[(in_idx, vertex)] for in_idx in in_idxs], axis=0)
            out_flow = cp.sum([y_vars[(vertex, out_idx)] for out_idx in out_idxs], axis=0)
            constraints += [in_flow == out_flow]
            # print(str(vertex) + "\tin: " + str(in_idxs) + "\tout: " + str(out_idxs))
            # print(str(vertex) + "\tin: " + str(in_flow) + "\tout: " + str(out_flow))

        # Convex Relaxation of Integer Constraints
        for edge in phi_vars.keys():
            phi = phi_vars[edge]
            constraints += [phi >= 0]
            constraints += [phi <= 1]

        # Construct and solve the convex relaxation
        prob = cp.Problem(cp.Minimize(objective), constraints)
        prob.solve()

        if prob.value == np.inf:
            print("Problem is infeasible!")
            return []

        # print("Final edge flows:")
        # for edge in phi_vars.keys():
        # 	print(str(edge) + "\t" + str(phi_vars[edge].value))

        # print("Final vertex flows:")
        # for vertex in range(len(adj_mat)):
        # 	print(str(vertex) + "\tin: " + str(v_in_flows[vertex].value) + "\tout: " + str(v_out_flows[vertex].value))

        # Reconstruct the x variables
        x = np.zeros((len(adj_mat), 2))
        out_idxs = np.nonzero(adj_mat[start_idx, :])[0]

        for v in range(len(adj_mat)):
            out_idxs = np.nonzero(adj_mat[v, :])[0]
            in_idxs = np.nonzero(adj_mat[:, goal_idx])[0]
            if v == start_idx:
                x[v] = np.sum([y_vars[(start_idx, out_idx)].value for out_idx in out_idxs], axis=0)
            elif v == goal_idx:
                x[v] = np.sum([z_vars[(in_idx, goal_idx)].value for in_idx in in_idxs], axis=0)
            else:
                x[v] = np.sum([y_vars[(v, out_idx)].value for out_idx in out_idxs], axis=0)

        # print("Final vertex positions")
        # for v in range(len(x)):
        # 	print(str(v) + "\t" + str(x[v]))

        # Deterministic rounded depth-first search
        # (TODO: Add in the randomized version?)
        shortest_path_idx = []
        visited_idx = set()
        shortest_path_idx.append(start_idx)
        visited_idx.add(start_idx)
        while shortest_path_idx[-1] != goal_idx:
            curr_idx = shortest_path_idx[-1]
            adj_verts = np.nonzero(adj_mat[curr_idx])[0]
            adj_weights = np.array([phi_vars[(curr_idx, adj_vert)].value for adj_vert in adj_verts])
            adj_verts_sorted = adj_verts[np.argsort(-adj_weights)]
            adj_verts_open = np.array([adj_vert not in visited_idx for adj_vert in adj_verts_sorted])
            if np.sum(adj_verts_open) == 0:
                shortest_path_idx.pop()
            else:
                next_idx = adj_verts_sorted[np.nonzero(adj_verts_open)[0][0]]
                shortest_path_idx.append(next_idx)
                visited_idx.add(next_idx)

        shortest_path = [x[idx] for idx in shortest_path_idx]
        return shortest_path

    def draw_output_gcs(self, shortest_path, gcs_regions, region_points, conjugate_graph):
        fig, ax = plt.subplots()
        # ax.set_xlim(world_bounds[0])
        # ax.set_ylim(world_bounds[1])
        # ax.add_patch(Polygon(obstacle, color="red"))

        ax.add_patch(Circle(self.start, radius=0.1, color="green"))
        ax.add_patch(Circle(self.goal, radius=0.1, color="blue"))

        for idx, halfspace_rep in enumerate(gcs_regions):
            if halfspace_rep == "start" or halfspace_rep == "goal":
                continue
            color = plt.get_cmap("Set3")(float(idx) / 12.0)
            # draw_halfspace_rep(ax, halfspace_rep, color=color)

        # draw the regions
        ax.scatter(region_points[:, 0], region_points[:, 1], color="grey")
        for i in range(conjugate_graph.shape[0]):
            for j in range(i, conjugate_graph.shape[1]):
                if conjugate_graph[i, j]:
                    ax.plot(region_points[[i, j], 0], region_points[[i, j], 1], color="grey", linestyle="dashed")
            ax.text(region_points[i, 0] - 0.2, region_points[i, 1], str(i), color="grey")

        # draw the resulting path
        if len(shortest_path) > 0:
            shortest_path = np.asarray(shortest_path)
            ax.plot(shortest_path[:, 0], shortest_path[:, 1], color="black", lw=2)
            ax.scatter(shortest_path[:, 0], shortest_path[:, 1], color="black", lw=2)

        ax.set_aspect("equal")
        ax.set_title("GCS Solution")
        plt.show()


if __name__ == "__main__":
    from iris import Iris

    rand_obstacles = False
    rand_seed_points = False
    space_bound = [[0, 4], [0, 4]]
    iris = Iris(rand_obstacles, rand_seed_points, space_bound)
    region_tuples = iris.get_region_tuples()
    halfspace_reps = iris.get_halfspace(region_tuples)
    iris.draw_output_iris(halfspace_reps)

    gsc = GCS()
    overlap_adj_mat = gsc.construct_overlap_adj_mat(halfspace_reps)
    gcs_regions, region_points, gcs_regions_idx_dict, conjugate_graph = gsc.construct_gcs_regions(overlap_adj_mat, halfspace_reps)
    shortest_path = gsc.solve_gcs_rounding(gcs_regions, conjugate_graph)
    gsc.draw_output_gcs(shortest_path, gcs_regions, region_points, conjugate_graph)
