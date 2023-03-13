import os
import sys
wd = os.path.abspath(os.getcwd())
sys.path.append(str(wd))

import numpy as np
from map.load_map import grid_map_probability
import matplotlib.pyplot as plt

def Sampling(map):
    row = map.shape[0]

    p = np.ravel(map) / np.sum(map)

    x_sample = np.random.choice(len(p), p=p)

    x = x_sample // row
    y = x_sample % row

    x = np.random.uniform(low=x - 0.5, high=x + 0.5)
    y = np.random.uniform(low=y - 0.5, high=y + 0.5)

    x_rand = np.array([x, y])

    return x_rand

if __name__=="__main__":

    # Grid Sampling
    map1 = grid_map_probability(0, 0, False)
    map2 = grid_map_probability(0, 3, False)
    map3 = grid_map_probability(0, 3, True)

    plt.figure(figsize=(10,10))
    plt.axes().set_aspect('equal')
    for i in range(1000):
        x = np.random.uniform(low = 0, high = map1.shape[0])
        y = np.random.uniform(low = 0, high = map1.shape[1])
        plt.scatter(x, y, c="red", s=2)
        if i%100 == 0:
            print(i)

    plt.imshow(np.transpose(map1),cmap = "gray", interpolation = 'nearest')
    plt.show()

    plt.figure(figsize=(10,10))
    plt.axes().set_aspect('equal')
    for i in range(1000):
        x = Sampling(map1)
        plt.scatter(x[0], x[1], c="red", s=2)
        if i%100 == 0:
            print(i)

    plt.imshow(np.transpose(map1),cmap = "gray", interpolation = 'nearest')
    plt.show()

    plt.figure(figsize=(10,10))
    plt.axes().set_aspect('equal')
    for i in range(1000):
        x = Sampling(map2)
        plt.scatter(x[0], x[1], c="red", s=2)
        if i%100 == 0:
            print(i)

    plt.imshow(np.transpose(map2),cmap = "gray", interpolation = 'nearest')
    plt.show()

    plt.figure(figsize=(10,10))
    plt.axes().set_aspect('equal')
    for i in range(1000):
        x = Sampling(map3)
        plt.scatter(x[0], x[1], c="red", s=2)
        if i%100 == 0:
            print(i)

    plt.imshow(np.transpose(map3),cmap = "gray", interpolation = 'nearest')
    plt.show()