import os
import sys
wd= os.path.abspath(os.getcwd())
sys.path.append(str(wd))

import numpy as np
from skimage.measure import profile_line
import matplotlib.pyplot as plt
from map import taskmap_img_format

configuration_space = []
map = taskmap_img_format.map_2d_1()
plt.imshow(map)
plt.show()

profile_prob1 = profile_line(map, (7,0), (7,30), linewidth=2, order=0, reduce_func=None)
print("==>> profile_prob1: \n", profile_prob1)
plt.imshow(profile_prob1)
plt.show()

profile_prob2 = profile_line(map, (15,0), (15,30), linewidth=2, order=0, reduce_func=None)
print("==>> profile_prob2: \n", profile_prob2)
plt.imshow(profile_prob2)
plt.show()

profile_prob = np.concatenate((profile_prob1, profile_prob2))
print("==>> profile_prob: \n", profile_prob)
plt.imshow(profile_prob)
plt.show()

if val_temp:= (0 in profile_prob): # if the line has contact with hard obstacle (0 value) of the map, thus the config space will directly block(= assign 0 value to config space)
    prob = 0
    print(val_temp)
else: # if the line has no contact with hard obstacle (check if there is no 0), then assign the value to config space equal the the lowest probability in the list (free = 1 or soft obs = (1 to 0.3...))
    prob = np.min(profile_prob)
configuration_space.append([0, 0, prob])

print(configuration_space)