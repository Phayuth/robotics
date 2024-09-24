import os
import sys
sys.path.append(str(os.path.abspath(os.getcwd())))

import glob
import numpy as np
from scipy.ndimage import binary_dilation
from spatial_geometry.spatial_shape import ShapeRectangle


class CostMapLoader:

    def __init__(self, costmap) -> None:
        self.costmap = costmap

    @classmethod
    def loadarray(cls, costmap): # load from numpy array
        return cls(costmap)

    @classmethod
    def loadsave(cls, maptype, mapindex, reverse=False): # load from numpy save
        task_map_list = glob.glob('./datasave/task_space/*.npy')
        config_map_list = glob.glob('./datasave/config_space_data_2d/*.npy')
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
                cell_value = np.sum(self.costmap[i - size:i + size + 1, j - size:j + size + 1]) / ((2 * size + 1)**2)
                kernel_map = np.append(kernel_map, cell_value)
        self.costmap = np.reshape(kernel_map, (self.costmap.shape[0] - 2 * size, self.costmap.shape[1] - 2 * size))

    def grid_map_probability(self, size=3):
        self.increase_map_size(multiplier=4)
        self.obs_dilation()
        self.padding_four_size(size)
        self.probabilitizer(size)


class CostMapClass:

    def __init__(self, maploader: CostMapLoader, maprange: list = None) -> None:
        self.costmap = maploader.costmap
        if maprange is None:
            self.xmin, self.xmax = 0, self.costmap.shape[1]  # colum is x
            self.ymin, self.ymax = 0, self.costmap.shape[0]  # row is y
        else:
            self.xmin, self.xmax = maprange[0][0], maprange[0][1]
            self.ymin, self.ymax = maprange[1][0], maprange[1][1]

    def costmap2geo(self, free_space_value=1):
        size_x = self.costmap.shape[1]
        size_y = self.costmap.shape[0]

        xval = np.linspace(self.xmin, self.xmax, size_x)
        yval = np.linspace(self.ymin, self.ymax, size_y)

        obj = [ShapeRectangle(xval[i], yval[j], xval[i + 1] - xval[i], yval[j + 1] - yval[j], p=free_space_value) for i in range(len(xval) - 1) for j in range(len(yval) - 1) if self.costmap[j, i] != free_space_value]

        return obj


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from spatial_geometry.taskmap_img_format import map_2d_1

    # load from numpy array
    # loader = CostMapLoader.loadarray(map_2d_1())
    # plt.imshow(loader.costmap)
    # plt.show()

    # load from numpy save
    loader = CostMapLoader.loadsave(maptype="task", mapindex=0, reverse=False)
    plt.imshow(loader.costmap)
    plt.show()

    loader.grid_map_probability()
    plt.imshow(loader.costmap)
    plt.show()

    # create map class and probibilitize step by step
    map = CostMapClass(loader, maprange=[[-np.pi, np.pi], [-np.pi, np.pi]])
    print(map.xmin, map.xmax, map.ymin, map.ymax)
    plt.imshow(map.costmap)
    plt.show()

    # convert from image to geometry format
    obj_list = map.costmap2geo(free_space_value=1)
    for obs in obj_list:
        obs.plot()
    plt.show()