import numpy as np
import matplotlib.pyplot as plt

# data
siz = 20
voxelarray = np.ones((siz,siz,siz), dtype=bool)

# plot everything
ax = plt.figure().add_subplot(projection='3d')
ax.voxels(voxelarray,  edgecolor='k')

plt.show()