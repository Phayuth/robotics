import matplotlib.pyplot as plt
from pytransform3d.transformations import plot_transform
from pytransform3d.plot_utils import make_3d_axis
from pytransform3d.transformations import vectors_to_points
import numpy as np

p = 0.001*np.array(
    [
        [375.6, 350.05, -700.1],
        [358.8, -363.92, -695.1],
        [359.54, -309.85, -696.86],
        [362.8, -163.65, -697.15],
        [393.04, 203.9, -699.2],
        [-561.95, 359.95, -691.92],
        [-561.69, -352.84, -692.13],
    ]
)
l = 930
w = 720
ax = make_3d_axis(ax_s=1, unit="m", n_ticks=6)
plot_transform(ax=ax, name="base_link")
for i in range(p.shape[0]):
    ax.scatter(p[i, 0], p[i, 1], p[i, 2], s=10, c="r", label="Points")
    ax.text(p[i, 0], p[i, 1], p[i, 2], f"p{i}", fontsize=12)
ax.set_xlabel("X-axis (m)")
ax.set_ylabel("Y-axis (m)")
ax.set_zlabel("Z-axis (m)")
ax.legend()

plt.show()

pxy = p[:, :2]
from shapely.geometry.polygon import Polygon
from shapely import convex_hull, MultiPoint
mp = MultiPoint(pxy)
cx = convex_hull(mp)
print(cx)