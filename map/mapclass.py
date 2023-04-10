import numpy as np
import matplotlib.pyplot as plt
import glob
from scipy.ndimage import binary_dilation


class MapLoader:

    def __init__(self, costmap) -> None:
        self.costmap = costmap

    @classmethod
    def loadarray(cls, costmap):
        return cls(costmap)

    @classmethod
    def loadsave(cls, maptype, mapindex, reverse):
        task_map_list = glob.glob('./map/mapdata/task_space/*.npy')
        config_map_list = glob.glob('./map/mapdata/config_space_data_2d/*.npy')
        if maptype == "task":
            costmap = 1 - np.load(task_map_list[mapindex]).astype(np.uint8)
        elif maptype == "config":
            costmap = 1 - np.load(config_map_list[mapindex]).astype(np.uint8)
        if reverse:
            costmap = 1 - costmap
        return cls(costmap)

    def getcostmap(self):
        return self.costmap

    def increase_map_size(self, multiplier=4):
        self.costmap = np.repeat(self.costmap, multiplier, axis=1)
        self.costmap = np.repeat(self.costmap, multiplier, axis=0)

    def obs_dilation(self):
        self.costmap = binary_dilation(1 - self.costmap).astype(self.costmap.dtype)
        self.costmap = 1 - self.costmap

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
                print(i, j)
                cell_value = np.sum(self.costmap[i - size:i + size + 1, j - size:j + size + 1]) / ((2 * size + 1)**2)
                kernel_map = np.append(kernel_map, cell_value)
        self.costmap = np.reshape(kernel_map, (self.costmap.shape[0] - 2 * size, self.costmap.shape[1] - 2 * size))

    def grid_map_probability(self, size=3):
        self.increase_map_size(multiplier=4)
        self.obs_dilation()
        self.padding_four_size(size)
        self.probabilitizer(size)


class MapClass:

    def __init__(self, maploader: MapLoader, maprange: list = None) -> None:
        self.costmap = maploader.costmap
        if maprange is None:
            self.xmin, self.xmax = 0, self.costmap.shape[1]  # colum is x
            self.ymin, self.ymax = 0, self.costmap.shape[0]  # row is y
        else:
            self.xmin, self.xmax = maprange[0][0], maprange[0][1]
            self.ymin, self.ymax = maprange[1][0], maprange[1][1]


if __name__ == "__main__":
    import taskmap_img_format

    # SECTION - load from numpy array
    loader = MapLoader.loadarray(taskmap_img_format.map_2d_1())
    plt.imshow(loader.costmap)
    plt.show()

    # SECTION - load from numpy save
    loader = MapLoader.loadsave(maptype="task", mapindex=0, reverse=False)
    plt.imshow(loader.costmap)
    plt.show()

    # SECTION - create map class and probibilitize step by step
    map = MapClass(loader, maprange=[[-np.pi, np.pi], [-np.pi, np.pi]])
    print(map.xmin, map.xmax, map.ymin, map.ymax)
    plt.imshow(map.costmap)
    plt.show()