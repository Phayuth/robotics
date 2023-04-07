import numpy as np
import matplotlib.pyplot as plt
import glob
from scipy.ndimage import binary_dilation

task_map_list = glob.glob('./map/mapdata/task_space/*.npy')
config_map_list = glob.glob('./map/mapdata/config_space_data_2d/*.npy')


def costmaploader(maptype:str, mapindex:int, reverse:False):
    if maptype == "task":
        map = 1 - np.load(task_map_list[mapindex]).astype(np.uint8)
    elif maptype == "config":
        map = 1 - np.load(config_map_list[mapindex]).astype(np.uint8)

    if reverse:
        return map
    else:
        return 1 - map


class MapClass:
    def __init__(self, costmap:np.ndarray, maprange:list=None) -> None:
        self.costmap = costmap
        if maprange is None:
            self.xmin = 0
            self.xmax = self.costmap.shape[1] # colum is x
            self.ymin = 0
            self.ymax = self.costmap.shape[0] # row is y
        else:
            self.xmin = maprange[0][0]
            self.xmax = maprange[0][1]
            self.ymin = maprange[1][0]
            self.ymax = maprange[1][1]

    def increase_map_size(self, multiplier=4):
        self.costmap = np.repeat(self.costmap, multiplier, axis=1)
        self.costmap = np.repeat(self.costmap, multiplier, axis=0)

    def obs_dilation(self):
        self.costmap = binary_dilation(self.costmap).astype(self.costmap.dtype)

    def padding_four_size(self, size=3):
        f1 = np.zeros((self.costmap.shape[0], 1))
        for _ in range(size):
            self.costmap = np.hstack((self.costmap, f1))
            self.costmap = np.hstack((f1, self.costmap))

        f2 = np.zeros((1, self.costmap.shape[1]))
        for _ in range(size):
            self.costmap = np.vstack((self.costmap, f2))
            self.costmap = np.vstack((f2, self.costmap))

    def probabilitizer(self, size=3):
        kernel_map = np.array([])
        for i in range(size, self.costmap.shape[0] - size):
            for j in range(size, self.costmap.shape[1] - size):
                cell_value = np.sum(self.costmap[i - size:i + size + 1, j - size:j + size + 1]) / ((2 * size + 1) ** 2)
                kernel_map = np.append(kernel_map, cell_value)
        self.costmap = np.reshape(kernel_map, (self.costmap.shape[0] - 2 * size, self.costmap.shape[1] - 2 * size))

if __name__ == "__main__":
    map = MapClass(costmaploader("config", 1, False))
    map.increase_map_size(4)
    map.obs_dilation()
    map.padding_four_size()
    map.probabilitizer()
    plt.imshow(map.costmap)
    plt.show()