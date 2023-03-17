import numpy as np
from scipy.ndimage import binary_dilation

def increase_map_size(map, multiplier):
    map = np.repeat(map, multiplier, axis=1)
    map = np.repeat(map, multiplier, axis=0)
    return map

def obs_dilation(map):
    map = binary_dilation(map).astype(map.dtype)
    return map

if __name__=="__main__":
    import matplotlib.pyplot as plt
    import glob

    index = 0
    map_list = glob.glob('./map/mapdata/task_space/*.npy')
    
    map = np.load(map_list[index]).astype(np.uint8)
    plt.imshow(map)
    plt.show()

    map = increase_map_size(map, multiplier=5)
    plt.imshow(map)
    plt.show()

    map = obs_dilation(map)
    plt.imshow(map)
    plt.show()