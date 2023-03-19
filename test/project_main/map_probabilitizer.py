import numpy as np
import matplotlib.pyplot as plt
import glob
from scipy.ndimage import binary_dilation

map_list = glob.glob('./map/mapdata/task_space/*.npy')

def grid_map_probability(index, size, classify):
    # load map from task_space folder and convert to probability form
    map = np.load(map_list[index]).astype(np.uint8)
    map = np.repeat(map, 4, axis=1)
    map = np.repeat(map, 4, axis=0)

    map = binary_dilation(map).astype(map.dtype)

    f1 = np.zeros((map.shape[0], 1))
    for i in range(size):
        map = np.hstack((map, f1))
        map = np.hstack((f1, map))

    f2 = np.zeros((1, map.shape[1]))
    for i in range(size):
        map = np.vstack((map, f2))
        map = np.vstack((f2, map))

    kernel_map = np.array([])
    for i in range(size, map.shape[0] - size):
        for j in range(size, map.shape[1] - size):
            kernel_map = np.append(kernel_map, np.sum(map[i - size:i + size + 1, j - size:j + size + 1]) / ((2 * size + 1) ** 2))

    kernel_map = np.reshape(kernel_map, (map.shape[0] - 2 * size, map.shape[1] - 2 * size))

    return kernel_map

if __name__ == "__main__":
    index = 2
    map = np.load(map_list[index]).astype(np.float64)
    plt.imshow(map)
    plt.show()
    size = 4
    f1 = np.zeros((map.shape[0], 1))  # it just padding map with zero, all four size of map
    for i in range(size):
        map = np.hstack((map, f1))
        map = np.hstack((f1, map))
    plt.imshow(map)
    plt.show()


    f2 = np.zeros((1, map.shape[1])) # it just padding map with zero, all four size of map
    for i in range(size):
        map = np.vstack((map, f2))
        map = np.vstack((f2, map))
    plt.imshow(map)
    plt.show()

    kernel_map = np.array([])
    for i in range(size, map.shape[0] - size):
        for j in range(size, map.shape[1] - size):
            tt = np.sum(map[i - size:i + size + 1, j - size:j + size + 1]) / ((2 * size + 1) ** 2)
            kernel_map = np.append(kernel_map, tt)

    kernel_map = np.reshape(kernel_map, (map.shape[0] - 2 * size, map.shape[1] - 2 * size))

    plt.imshow(kernel_map)
    plt.show()
