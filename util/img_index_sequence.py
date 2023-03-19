import numpy as np
import matplotlib.pyplot as plt

array1 = np.array([[0,1],
                  [1,0]]).astype(int)

array2 = np.array([[0,0],
                   [1,1]])

array3 = np.array([[1,1],
                   [0,0]])

array4 = np.array([[1,0],
                   [0,1]])

fig, axs = plt.subplots(2, 2)
axs[0, 0].imshow(array1)
axs[0, 1].imshow(array2)
axs[1, 0].imshow(array3)
axs[1, 1].imshow(array4)
plt.show()

img_range = 0.1*np.arange(100).reshape(100,1)
plt.imshow(img_range)
plt.show()