import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

fig = plt.figure(figsize=(4,4))
ax = fig.add_subplot(111, projection='3d')

# data
x = [1,2,3,4,5,6,7,8,9,10]
y = [1,2,3,4,5,6,7,8,9,10]
z = [1,2,3,4,5,6,7,8,9,10]

data = np.random.random_sample((3,1000))

# plot line
# ax.plot3D(x, y, z, 'red')

# plot scatter
# ax.scatter(x, y, z, c=z, cmap='cividis')

# limit axis
# ax.set_xlim(-1, 1)
# ax.set_ylim(-1, 1)
# ax.set_zlim(-1, 1)

# set label name
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

ax.scatter(data[0], data[1], data[2], cmap='cividis')
plt.show()