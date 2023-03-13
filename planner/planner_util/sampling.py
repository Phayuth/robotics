import os
import sys
wd = os.path.abspath(os.getcwd())
sys.path.append(str(wd))

import numpy as np
from map.load_map import grid_map_probability
import matplotlib.pyplot as plt

def bias_sampling(map):

    row = map.shape[0]
    p = np.ravel(map) / np.sum(map)
    x_sample = np.random.choice(len(p), p=p)
    x = x_sample // row
    y = x_sample % row
    x = np.random.uniform(low=x - 0.5, high=x + 0.5)
    y = np.random.uniform(low=y - 0.5, high=y + 0.5)
    x_rand = np.array([x, y])

    return x_rand

def uniform_sampling(map):

    x = np.random.uniform(low=0, high=map.shape[0])
    y = np.random.uniform(low=0, high=map.shape[1])
    x_rand = np.array([x, y])

    return x_rand

def start_sampling(map, number_sampling, sampling_mode):
    plt.figure(figsize=(10,10))
    plt.axes().set_aspect('equal')

    for i in range(number_sampling):
        if sampling_mode =="bias":
            x = uniform_sampling(map)
        elif sampling_mode =="uniform":
            x = uniform_sampling(map)

        plt.scatter(x[0], x[1], c="red", s=2)

    plt.imshow(np.transpose(map),cmap = "gray", interpolation = 'nearest')
    plt.show()

if __name__=="__main__":

    map1 = grid_map_probability(0, 0, False)
    map2 = grid_map_probability(0, 3, False)
    map3 = grid_map_probability(0, 3, True)

    print("starting sampling")
    start_sampling(map3, number_sampling=1000, sampling_mode="uniform")