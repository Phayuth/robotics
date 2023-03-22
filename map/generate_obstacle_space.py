import os
import sys
wd = os.path.abspath(os.getcwd())
sys.path.append(str(wd))

from shapely.geometry import Polygon
import matplotlib.pyplot as plt
import numpy as np
import glob

map_list = glob.glob('./map/mapdata/task_space/*.npy')

def Obstacle_generater(obstacle):
    obs = []
    for o in obstacle:
        obs.append(Polygon(o))
    return obs

def Obstacle_center(obstacle):
    center = []
    for o in obstacle:
        c_x = (o[0][0] + o[2][0]) / 2
        c_y = (o[0][1] + o[2][1]) / 2
        c = [c_x, c_y]
        center.append(c)
    return center

def Collision_range(obstacle):
    range = []
    for o in obstacle:
        d = np.linalg.norm(np.array(o[2]) - np.array(o[0]))
        range.append(d/2)
    return range

def Draw_obs(obs):
    fig, axs = plt.subplots(figsize=(10, 10))
    for o in obs:
        x, y = o.exterior.xy
        axs.fill(x, y, fc="black", ec="none")

def obstacle_generate_from_map(index):
    map = np.load(map_list[index]).astype(np.uint8)
    obs_center = []
    obs = []
    for i in range(map.shape[0]):
        for j in range(map.shape[1]):
            if map[i][j] == 1:
                obs_center.append([i, j])
                obs.append(((i-0.5, j-0.5), (i+0.5, j-0.5), (i+0.5, j+0.5), (i-0.5, j+0.5), (i-0.5, j-0.5)))
    return map, obs, obs_center

def obstacle_generate_from_map_rgb(imgArray):
    obs_center = []
    obs = []
    for i in range(imgArray.shape[0]):
        for j in range(imgArray.shape[1]):
            if imgArray[i][j][0] != 0 or imgArray[i][j][1] != 0 or imgArray[i][j][2] != 0:
                obs_center.append([j, i])
                obs.append(((j-0.5, i-0.5), (j+0.5, i-0.5), (j+0.5, i+0.5), (j-0.5, i+0.5), (j-0.5, i-0.5)))
    return imgArray, obs, obs_center


if __name__ == "__main__":
    
    from taskmap_img_format import bmap, pmap

    map, obstacle, obstacle_center = obstacle_generate_from_map(index=0)
    obs = Obstacle_generater(obstacle)
    collision_range = (2**(1/2))/2

    plt.imshow(map)
    plt.show()
    Draw_obs(obs)
    plt.show()

    rgb_map = bmap(return_rgb=True)
    map, obstacle, obstacle_center = obstacle_generate_from_map_rgb(rgb_map)
    obs = Obstacle_generater(obstacle)

    plt.imshow(rgb_map)
    plt.show()
    Draw_obs(obs)
    plt.show()
