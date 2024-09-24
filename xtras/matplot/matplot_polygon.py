import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull

def extract_boundary_points(xy_points):
    points = np.array(xy_points)
    hull = ConvexHull(points)
    boundary_indices = hull.vertices
    boundary_points = points[boundary_indices]
    return boundary_points

polygon1_points = [(1, 2), (3, 6), (5, 3), (8, 1)]
polygon2_points = [(2, 7), (4, 9), (7, 8), (6, 6)]
boundary_points1 = extract_boundary_points(polygon1_points)
boundary_points2 = extract_boundary_points(polygon2_points)

fig, ax = plt.subplots()
boundary_x1, boundary_y1 = zip(*boundary_points1)
boundary_x2, boundary_y2 = zip(*boundary_points2)
boundary_polygon1 = plt.Polygon(boundary_points1, edgecolor='red', facecolor='gray')
boundary_polygon2 = plt.Polygon(boundary_points2, edgecolor='green', facecolor='gray')
ax.add_patch(boundary_polygon1)
ax.add_patch(boundary_polygon2)
x_coords1, y_coords1 = zip(*polygon1_points)
x_coords2, y_coords2 = zip(*polygon2_points)
plt.scatter(x_coords1, y_coords1, label='Polygon 1 Points', color='blue')
plt.scatter(x_coords2, y_coords2, label='Polygon 2 Points', color='purple')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Extracted Boundary Polygons')
plt.legend()
plt.grid()
plt.show()
