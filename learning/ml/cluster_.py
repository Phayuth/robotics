import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from scipy.spatial import Delaunay


def rearrange_boundary_points(points, center_point):
    angles = np.arctan2(points[:, 1] - center_point[1], points[:, 0] - center_point[0])
    sorted_indices = np.argsort(angles)
    sorted_points = points[sorted_indices]
    return sorted_points


np.random.seed(0)
num_points = 100
xy_points = np.random.rand(num_points, 2) * 10

dbscan = DBSCAN(eps=0.5, min_samples=5)
cluster_labels = dbscan.fit_predict(xy_points)
num_clusters = len(np.unique(cluster_labels))
cluster_lists = [[] for _ in range(num_clusters)]
for i, label in enumerate(cluster_labels):
    cluster_lists[label].append(xy_points[i])

fig, ax = plt.subplots()
colors = plt.cm.Spectral(np.linspace(0, 1, num_clusters))
for i, cluster in enumerate(cluster_lists):
    cluster = np.array(cluster)
    hull = Delaunay(cluster)
    boundary_indices = np.unique(hull.convex_hull.flat)
    boundary_points = cluster[boundary_indices]

    center_point = np.mean(boundary_points, axis=0)
    sorted_boundary = rearrange_boundary_points(boundary_points, center_point)

    cluster_polygon = plt.Polygon(sorted_boundary, edgecolor=colors[i], facecolor="none")
    ax.add_patch(cluster_polygon)

    plt.scatter(cluster[:, 0], cluster[:, 1], color=colors[i], label=f"Cluster {i + 1}")

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_title("DBScan Clustering with Ordered Boundary Polygons")
ax.legend()
ax.grid()
plt.show()
