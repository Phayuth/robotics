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

def increase_map_size(map, multiplier):
    map = np.repeat(map, multiplier, axis=1)
    map = np.repeat(map, multiplier, axis=0)
    return map

def obs_dilation(map):
    map = binary_dilation(map).astype(map.dtype)
    return map

def padding_four_size(map, size):
    f1 = np.zeros((map.shape[0], 1))
    for _ in range(size):
        map = np.hstack((map, f1))
        map = np.hstack((f1, map))

    f2 = np.zeros((1, map.shape[1]))
    for _ in range(size):
        map = np.vstack((map, f2))
        map = np.vstack((f2, map))
    return map

def probabilitizer(map, size):
    # I think this function is to calculate obstacle distance (distance from current cell to near obstacle)
    # load map and convert map from binary to probability
    # map = 1 - map # in case of the value is switch
    kernel_map = np.array([])
    for i in range(size, map.shape[0] - size):
        for j in range(size, map.shape[1] - size):
            cell_value = np.sum(map[i - size:i + size + 1, j - size:j + size + 1]) / ((2 * size + 1) ** 2)
            kernel_map = np.append(kernel_map, cell_value)
    kernel_map = np.reshape(kernel_map, (map.shape[0] - 2 * size, map.shape[1] - 2 * size))
    return kernel_map

def grid_map_probability(index, size):
    map = np.load(map_list[index]).astype(np.uint8)
    map = increase_map_size(map, 4)
    map = obs_dilation(map)
    map = 1 - map
    map = padding_four_size(map, size)
    map = probabilitizer(map, size)
    return map

if __name__ == "__main__":

    # SECTION - load binary map
    index = 2
    map = grid_map_binary(index)
    plt.title("Original Map")
    plt.imshow(map)
    plt.show()


    # SECTION - increase map size
    map = increase_map_size(map, multiplier=4)
    plt.title("Increase Map size")
    plt.imshow(map)
    plt.show()


    # SECTION - dilation of obstacle
    map = obs_dilation(1- map)
    plt.title("Dilation Map")
    plt.imshow(map)
    plt.show()


    # SECTION - probabilitize map
    map = probabilitizer(map, size=3)
    plt.title("Probability Map")
    plt.imshow(map)
    plt.show()


    # SECTION - Full set of all operation above in one function
    filter_size = 3
    map = grid_map_probability(index, filter_size)
    plt.imshow(map)
    plt.show()
