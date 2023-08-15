# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# import numpy as np

# fig = plt.figure(figsize=(4,4))
# ax = fig.add_subplot(111, projection='3d')

# # data
# x = [1,2,3,4,5,6,7,8,9,10]
# y = [1,2,3,4,5,6,7,8,9,10]
# z = [1,2,3,4,5,6,7,8,9,10]

# data = np.random.random_sample((3,1000))

# # plot line
# # ax.plot3D(x, y, z, 'red')

# # plot scatter
# # ax.scatter(x, y, z, c=z, cmap='cividis')

# # limit axis
# # ax.set_xlim(-1, 1)
# # ax.set_ylim(-1, 1)
# # ax.set_zlim(-1, 1)

# # set label name
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')

# ax.scatter(data[0], data[1], data[2], cmap='cividis')
# plt.show()

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from scipy.spatial import Delaunay

def rearrange_boundary_points(points, center_point):
    angles = np.arctan2(points[:, 1] - center_point[1], points[:, 0] - center_point[0])
    sorted_indices = np.argsort(angles)
    sorted_points = points[sorted_indices]
    return sorted_points

# Generate random XY points for demonstration
np.random.seed(0)
num_points = 100
xy_points = np.random.rand(num_points, 2) * 10

# Perform DBScan clustering
dbscan = DBSCAN(eps=0.5, min_samples=5)
cluster_labels = dbscan.fit_predict(xy_points)

# Separate points into different lists based on clusters
num_clusters = len(np.unique(cluster_labels))
cluster_lists = [[] for _ in range(num_clusters)]
for i, label in enumerate(cluster_labels):
    cluster_lists[label].append(xy_points[i])

# Create a figure and axis
fig, ax = plt.subplots()

# Plot the clustered points and polygons around cluster boundaries
colors = plt.cm.Spectral(np.linspace(0, 1, num_clusters))
for i, cluster in enumerate(cluster_lists):
    cluster = np.array(cluster)
    hull = Delaunay(cluster)
    boundary_indices = np.unique(hull.convex_hull.flat)
    boundary_points = cluster[boundary_indices]

    # Choose a center point for sorting the boundary points
    center_point = np.mean(boundary_points, axis=0)
    sorted_boundary = rearrange_boundary_points(boundary_points, center_point)

    cluster_polygon = plt.Polygon(sorted_boundary, edgecolor=colors[i], facecolor='none')
    ax.add_patch(cluster_polygon)

    plt.scatter(cluster[:, 0], cluster[:, 1], color=colors[i], label=f'Cluster {i + 1}')

plt.xlabel('X')
plt.ylabel('Y')
plt.title('DBScan Clustering with Ordered Boundary Polygons')
plt.legend()
plt.grid()

# Show the plot
plt.show()
