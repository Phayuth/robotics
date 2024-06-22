import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.spatial import ConvexHull

# plt.rcParams.update({
#     "font.family": "serif",  # use serif/main font for text elements
#     "text.usetex": True,     # use inline math for ticks
#     "pgf.rcfonts": False     # don't setup fonts from rc parameters
#     })

# Generate random XY points for demonstration
np.random.seed(0)
num_points = 100
xy_points = np.random.rand(num_points, 2) * 10

# Number of clusters
num_clusters = 3

# Perform K-Means clustering
kmeans = KMeans(n_clusters=num_clusters)
cluster_labels = kmeans.fit_predict(xy_points)

# Separate points into different lists based on clusters
cluster_lists = [[] for _ in range(num_clusters)]
for i, label in enumerate(cluster_labels):
    cluster_lists[label].append(xy_points[i])

# Create a figure and axis
fig, ax = plt.subplots()
fig.set_size_inches(w=3.40067, h=3.40067)

# Plot the clustered points and convex hulls
colors = ['red', 'green', 'blue']
for i, cluster in enumerate(cluster_lists):
    cluster = np.array(cluster)
    hull = ConvexHull(cluster)
    boundary_points = cluster[hull.vertices]

    cluster_polygon = plt.Polygon(boundary_points, edgecolor=colors[i], facecolor='none')
    ax.add_patch(cluster_polygon)

    plt.scatter(cluster[:, 0], cluster[:, 1], color=colors[i], label=f'Cluster {i + 1}')

plt.xlabel('X')
plt.ylabel('Y')
plt.title('Clustered Points with Convex Hulls')
plt.legend()
plt.grid()

# Set aspect ratio to be equal and limit axis ranges
ax.set_aspect('equal', adjustable='datalim')
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)

# Show the plot
plt.show()