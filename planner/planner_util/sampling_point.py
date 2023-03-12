import numpy as np
import random
import matplotlib.pyplot as plt

def sample_a_random_point_in_2d_distance_from_base_point(x_base,y_base,dist):
    
    x_sample = random.randint(-10, 10)
    y_sample = random.randint(-10, 10)

    theta = np.arctan2((y_sample - y_base),(x_sample - x_base))
    x_new = dist*np.cos(theta)
    y_new = dist*np.sin(theta)
    
    return x_new, y_new, x_sample, y_sample

x_base = 0
y_base = 0
x_new, y_new, x_sample, y_sample = sample_a_random_point_in_2d_distance_from_base_point(x_base,y_base,dist=2)

print(x_new, y_new)
print(x_sample, y_sample)
plt.scatter(0, 0)
plt.scatter(x_new, y_new)
plt.scatter(x_sample, y_sample)
plt.show()