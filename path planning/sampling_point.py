import numpy as np
import random
import matplotlib.pyplot as plt

def sample_a_point(x_base,y_base,d):
    x_sample = random.randint(-10, 10)
    y_sample = random.randint(-10, 10)

    theta = np.arctan((y_sample - y_base)/(x_sample - x_base))
    x_new = d*np.cos(theta)
    y_new = d*np.sin(theta)
    
    return x_new, y_new, x_sample, y_sample

x = 0
y = 0
dist = 2

x_new, y_new, x_sample, y_sample = sample_a_point(x,y,dist)

d = [(x_new - x),(y_new - y)]

norm = np.linalg.norm(d)
print(norm)

plt.scatter(x, y)
plt.scatter(x_new, y_new)
plt.scatter(x_sample, y_sample)
plt.show()