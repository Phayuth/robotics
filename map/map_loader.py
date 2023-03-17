import numpy as np
import matplotlib.pyplot as plt
import glob
from scipy.ndimage import binary_dilation

map_list = glob.glob('./map/mapdata/task_space/*.npy')

def grid_map_binary(index):
    # load map from task_space folder in binary form
    map = np.load(map_list[index]).astype(np.uint8)
    map = 1 - map
    return map

def grid_map_probability(index, size, classify):
    # load map from task_space folder and convert to probability form
    map = np.load(map_list[index]).astype(np.uint8)
    map = np.repeat(map, 4, axis=1)
    map = np.repeat(map, 4, axis=0)

    map = binary_dilation(map).astype(map.dtype)

    if classify == True:
        for i in range(30, 80):
            for j in range(60, 100):
                map[i, j] = 0.2 * map[i, j]

        for i in range(80,100):
            for j in range(40, 60):
                map[i, j] = 0.2 * map[i, j]

        for i in range(90,110):
            for j in range(17, 40):
                map[i, j] = 0.2 * map[i, j]

    map = 1 - map

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
    # load binary map
    index = 2
    map = grid_map_binary(index)
    plt.imshow(map)
    plt.show()

    # Load Probability map conversion
    filter_size = 3 # 1 = 3x3, 2 = 5x5, 3 = 7x7
    classify = True
    map = grid_map_probability(index,filter_size,classify)
    plt.imshow(map)
    plt.show()
