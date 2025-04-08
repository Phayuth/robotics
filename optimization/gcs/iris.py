import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Circle
import cvxpy as cp
from scipy.spatial import HalfspaceIntersection
from shapely.geometry import LineString, Point
from shapely.geometry import Polygon as SPolygon
from shapely.geometry import MultiPolygon as SMultiPolygon
from scipy.optimize import linprog
import alphashape
from scipy.spatial import Voronoi, voronoi_plot_2d, ConvexHull, Delaunay

np.random.seed(0)


class Iris:

    def __init__(self, obs_mode, seed_mode, space_bound=[[0, 1], [0, 1]]):
        self.space_bound = np.array(space_bound)

        if obs_mode == "fixed":
            self.tris = self.fixed_obstacles()
        elif obs_mode == "random":
            self.tris = self.random_obstacles()

        if seed_mode == "fixed":
            self.iris_seed_points = self.fixed_iris_seed_points()
        elif seed_mode == "grid":
            self.iris_seed_points = self.grid_iris_seed_points()
        elif seed_mode == "random":
            self.iris_seed_points = self.random_iris_seed_points(n_points=15)
        elif seed_mode == "voronoi":
            self.iris_seed_points = self.random_voronoi_iris_seed_points(n_points=15)

        self.tolerance = 0.00001
        self.max_iters = 10

    def fixed_obstacles(self):
        # compose of triangles 3 points each of xy coordinates
        tris = [
            # main obstacles
            [[1, 2], [3, 2], [2, 1]],  # Obstacle Bottom
            [[1, 2], [3, 2], [2, 3]],  # Obstacle Top
            # put obstacles to close the space
            [[0, 0], [4, 0], [2, -1]],  # South
            [[0, 0], [0, 4], [-1, 2]],  # West
            [[0, 4], [4, 4], [2, 5]],  # North
            [[4, 4], [4, 0], [5, 2]],  # East
        ]
        return tris

    def random_obstacles(self):
        n_points = 200
        alpha = 25.0 / 4

        points = np.random.random(size=(n_points, 2)) * 4
        gen = alphashape.alphasimplices(points)

        tris = []
        for simplex, r in gen:
            if r < 1 / alpha:
                tris.append(points[simplex])

        # put obstacles to close the space
        tris.append(np.array([[0, 0], [4, 0], [2, -1]]))
        tris.append(np.array([[0, 0], [0, 4], [-1, 2]]))
        tris.append(np.array([[0, 4], [4, 4], [2, 5]]))
        tris.append(np.array([[4, 4], [4, 0], [5, 2]]))

        return tris

    def fixed_iris_seed_points(self):
        iris_seed_points = np.array([[1, 1], [3, 1], [1, 3], [3, 3]])
        return iris_seed_points

    def grid_iris_seed_points(self):
        pass

    def random_iris_seed_points(self, n_points):
        iris_seed_points = []
        poly = [SPolygon(p) for p in self.tris]
        polygon = SMultiPolygon(poly)
        while len(iris_seed_points) < n_points:
            # point = np.multiply(np.random.rand(2), self.space_bound[:, 1])
            point = np.random.uniform(self.space_bound[:, 0], self.space_bound[:, 1])
            if not polygon.contains(Point(point[0], point[1])):
                iris_seed_points.append(point)
        return np.array(iris_seed_points)

    def random_voronoi_iris_seed_points(self, n_points):
        # point = np.random.uniform(self.space_bound[:, 0], self.space_bound[:, 1], size=n_points)
        # vor = Voronoi(point)
        # iris_seed_points = vor.vertices
        # return iris_seed_points
        pass

    def ClosestPointOnObstacle(self, C, C_inv, d, o):
        v_tildes = C_inv @ (o - d).T
        n = 2
        m = len(o)
        x_tilde = cp.Variable(n)
        w = cp.Variable(m)
        prob = cp.Problem(cp.Minimize(cp.sum_squares(x_tilde)), [v_tildes @ w == x_tilde, w @ np.ones(m) == 1, w >= 0])
        prob.solve()
        x_tilde_star = x_tilde.value
        dist = np.sqrt(prob.value) - 1
        x_star = C @ x_tilde_star + d
        return x_star, dist

    def TangentPlane(self, C, C_inv2, d, x_star):
        a = 2 * C_inv2 @ (x_star - d).reshape(-1, 1)
        b = np.dot(a.flatten(), x_star)
        return a, b

    def SeparatingHyperplanes(self, C, d, O):
        C_inv = np.linalg.inv(C)
        C_inv2 = C_inv @ C_inv.T
        O_excluded = []
        O_remaining = O
        ais = []
        bis = []
        while len(O_remaining) > 0:
            obs_dists = np.array([np.min([np.linalg.norm(corner - d) for corner in o]) for o in O_remaining])
            best_idx = np.argmin(obs_dists)
            x_star, _ = self.ClosestPointOnObstacle(C, C_inv, d, O_remaining[best_idx])
            ai, bi = self.TangentPlane(C, C_inv2, d, x_star)
            ais.append(ai)
            bis.append(bi)
            idx_list = []
            for i, li in enumerate(O_remaining):
                redundant = [np.dot(ai.flatten(), xj) >= bi for xj in li]
                if i == best_idx or np.all(redundant):
                    idx_list.append(i)
            for i in reversed(idx_list):
                O_excluded.append(O_remaining[i])
                O_remaining.pop(i)
        A = np.array(ais).T[0]
        b = np.array(bis).reshape(-1, 1)
        return (A, b)

    def InscribedEllipsoid(self, A, b):
        n = 2
        C = cp.Variable((n, n), symmetric=True)
        d = cp.Variable(n)
        constraints = [C >> 0]
        constraints += [cp.atoms.norm2(ai.T @ C) + (ai.T @ d) <= bi for ai, bi in zip(A.T, b)]
        prob = cp.Problem(cp.Maximize(cp.atoms.log_det(C)), constraints)
        prob.solve()
        return C.value, d.value

    def solve_iris_region(self, seed_point):
        As = []
        bs = []
        Cs = []
        ds = []

        C0 = np.eye(2) * 0.01
        Cs.append(C0)
        ds.append(seed_point.copy())
        O = self.tris

        iters = 0
        while True:
            A, b = self.SeparatingHyperplanes(Cs[-1], ds[-1], O.copy())
            if np.any(A.T @ seed_point >= b.flatten()):
                # print("Terminating early to keep seed point in region.")
                break

            As.append(A)
            bs.append(b)

            C, d = self.InscribedEllipsoid(As[-1], bs[-1])
            Cs.append(C)
            ds.append(d)

            iters += 1

            if (np.linalg.det(Cs[-1]) - np.linalg.det(Cs[-2])) / np.linalg.det(Cs[-2]) < self.tolerance:
                break

            if iters > self.max_iters:
                break

        return As[-1], bs[-1], Cs[-1], ds[-1]

    def get_region_tuples(self):
        region_tuples = [self.solve_iris_region(seed_point) for seed_point in self.iris_seed_points]
        return region_tuples

    def compute_halfspace(self, A, b, d):
        ineq = np.hstack((A.T, -b))
        hs = HalfspaceIntersection(ineq, d, incremental=False)
        return hs

    def get_halfspace(self, region_tuples):
        halfspace_reps = [self.compute_halfspace(A, b, d) for A, b, _, d, in region_tuples]
        return halfspace_reps

    def order_vertices(self, points):
        center = np.mean(points, axis=0)
        centered_points = points - center
        thetas = np.arctan2(centered_points[:, 1], centered_points[:, 0])
        idxs = np.argsort(thetas)
        return points[idxs]

    def draw_halfspace_rep(self, ax, halfspace_rep, color):
        points = halfspace_rep.intersections
        current_region = self.order_vertices(points)
        ax.add_patch(Polygon(current_region, color=color, alpha=0.25))
        ax.plot(current_region[:, 0], current_region[:, 1], color=color, alpha=0.75)
        ax.plot(current_region[[0, -1], 0], current_region[[0, -1], 1], color=color, alpha=0.75)

    def draw_output_iris(self, halfspace_reps_list):
        fig, ax = plt.subplots()
        ax.set_xlim(self.space_bound[0])
        ax.set_ylim(self.space_bound[1])

        for tri in self.tris:
            ax.add_patch(Polygon(tri, color="red"))
        ax.scatter(self.iris_seed_points[:, 0], self.iris_seed_points[:, 1], color="grey")

        for idx, halfspace_rep in enumerate(halfspace_reps_list):
            if halfspace_rep == "start" or halfspace_rep == "goal":
                continue
            color = plt.get_cmap("Set3")(float(idx) / 12.0)
            self.draw_halfspace_rep(ax, halfspace_rep, color=color)

        ax.set_aspect("equal")
        ax.set_title("IRIS Regions")
        plt.show()


if __name__ == "__main__":
    obs_mode = "random"
    seed_mode = "random"
    space_bound = [[0, 4], [0, 4]]
    iris = Iris(obs_mode, seed_mode, space_bound)

    region_tuples = iris.get_region_tuples()
    halfspace_reps = iris.get_halfspace(region_tuples)
    iris.draw_output_iris(halfspace_reps)