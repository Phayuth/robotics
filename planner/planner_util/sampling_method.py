import os
import sys
wd = os.path.abspath(os.getcwd())
sys.path.append(str(wd))

import numpy as np
from map.map_loader import grid_map_probability
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


def unit_disc_sampling(map):

    r = np.random.uniform(low=0, high=1)
    phi = np.random.uniform(0, 2*np.pi)
    x = r * np.cos(phi) + map.shape[0]/2
    y = r * np.sin(phi) + map.shape[1]/2
    x_rand = np.array([x,y])

    return x_rand


def unit_nball_sampling(n = 2):

    u = np.random.normal(0, 1, n + 1) # sample on a unit n+1 sphere
    norm = np.linalg.norm(u)
    u = u/norm
    x_rand = u[:n]                    # the first n coordinates are uniform in a unit n ball
    
    return x_rand


def sphere_sampling(center, radius, n = 2):

    xball = unit_nball_sampling(n)
    x_rand = radius*xball + center

    return x_rand

    
def ellipsoid_sampling(center, n=2, axes=[], rot=[]):

    if len(rot) == 0: 
        rot = np.eye(n)
    if len(axes) == 0: # axes length across each dimension in ellipsoid frame
        axes = [1]*n 
    l = np.diag(axes)

    xball = unit_nball_sampling(n)
    x_rand = (rot @ l @ xball.T).T + center # transform points in unitball to ellipsoid # centre of the n-dimensional ellipsoid in the n-dimensional
    
    return x_rand


class informed_sampling:
    def __init__(self, goal, start, n=2):
        self.n = n
        self.cmin = np.linalg.norm(goal - start)
        self.center = (goal + start)/2
        self.c = self.rotationtoworldframe(goal, start)           #rotation matrix from ellipsoid frame to world frames

    def sample(self, cmax):
        #hyperspheroid axes lengths
        r1 = cmax/2 # cmax - current best cost
        ri = np.sqrt(cmax**2 - self.cmin**2)/2
        axes = [r1]+ [ri]*(self.n - 1)
        return ellipsoid_sampling(self.center, axes=axes, rot=self.c)

    def rotationtoworldframe(self, goal, start): # given two focal points goal and start in n-dimensions returns rotation matrix from the ellipsoid frame to the world frame
        e1 = (goal - start) / self.cmin                 # transverse axis of the ellipsoid in the world frame
        w1 = [1]+[0]*(self.n - 1)                       # first basis vector of the world frame [1,0,0,...]
        m = np.outer(e1,w1)                             # outer product of e1 and w1
        u, s, v = np.linalg.svd(m)                      # svd decomposition od outer product
        middlem = np.eye(self.n)                        # calculate the middle diagonal matrix
        middlem[-1,-1] = np.linalg.det(u)*np.linalg.det(v)
        c = u @ middlem @ v.T                           # calculate the rotation matrix

        return c


def start_sampling(map, number_sampling, sampling_mode):
    plt.figure(figsize=(10,10))
    plt.axes().set_aspect('equal')

    for i in range(number_sampling):
        if sampling_mode =="bias":
            x = uniform_sampling(map)
        elif sampling_mode =="uniform":
            x = uniform_sampling(map)
        elif sampling_mode =="unitdisc":
            x = unit_disc_sampling(map)
        elif sampling_mode =="unitnball":
            x = unit_nball_sampling(n=2)
        elif sampling_mode =="sphere":
            x = sphere_sampling(center=np.array([50,50]), radius=5, n = 2)
        elif sampling_mode =="ellip":
            x = ellipsoid_sampling(center=np.array([50,50]), axes=[3,7])
        elif sampling_mode == "infsamp":
            x = informed_sampling(np.array([50,50]), np.array([30,50])).sample(100)
        plt.scatter(x[0], x[1], c="red", s=2)

    plt.imshow(np.transpose(map),cmap = "gray", interpolation = 'nearest')
    plt.show()

if __name__=="__main__":

    map1 = grid_map_probability(0, 0)
    map2 = grid_map_probability(0, 3)
    map3 = grid_map_probability(0, 3)

    # start_sampling(map3, number_sampling=1000, sampling_mode="uniform")



    # SECTION - Test code
    row = map3.shape[0]
    p = np.ravel(map3) / np.sum(map3)
    x_sample = np.random.choice(len(p), p=p)
    x = x_sample // row
    y = x_sample % row
    x = np.random.uniform(low=x - 0.5, high=x + 0.5)
    y = np.random.uniform(low=y - 0.5, high=y + 0.5)
