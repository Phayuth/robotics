import numpy as np

def Sampling(map): # bias sampling
    row = map.shape[0]

    p = np.ravel(map) / np.sum(map)

    x_sample = np.random.choice(len(p), p=p)

    x = x_sample // row
    y = x_sample % row

    x = np.random.uniform(low=x - 0.5, high=x + 0.5)
    y = np.random.uniform(low=y - 0.5, high=y + 0.5)

    x_rand = np.array([x, y])

    return x_rand
