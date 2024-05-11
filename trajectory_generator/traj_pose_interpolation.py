import os
import sys

wd = os.path.abspath(os.getcwd())
sys.path.append(str(wd))

import numpy as np
from scipy.spatial.transform import Rotation, Slerp
from spatial_geometry.spatial_transformation import RigidBodyTransformation as rbt

start_pos = [1, 1, 1]
end_pos = [2, 2, 2]
numpose = 10
positions = np.linspace(start_pos, end_pos, numpose)


key_rots = Rotation.from_quat([[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 1.0, 0.0]])
key_times = [0, 1]
slerp = Slerp(key_times, key_rots)

times = np.linspace(0, 1, numpose)
interp_rots = slerp(times)

b = interp_rots.as_quat()


import matplotlib.pyplot as plt
from pytransform3d.transformations import plot_transform
from pytransform3d.plot_utils import make_3d_axis

ax = make_3d_axis(ax_s=1, unit="m", n_ticks=6)
plot_transform(ax=ax, name="base_link")
for i in range(numpose):
    H = rbt.conv_t_and_quat_to_h(positions[i], b[i])
    plot_transform(ax, A2B=H, s=0.1)

plt.tight_layout()
plt.show()
