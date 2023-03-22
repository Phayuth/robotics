import numpy as np
from skimage.measure import profile_line
    

if __name__=="__main__":
    import os
    import sys
    wd = os.path.abspath(os.getcwd())
    sys.path.append(str(wd))
    
    from map import taskmap_img_format
    import matplotlib.pyplot as plt

    configuration_space = []

    # Create map
    map = taskmap_img_format.map_2d_1()
    grid = 10
    theta1, theta2 = np.linspace(0, 360, grid), np.linspace(0, 360, grid)

    # for i in theta1:
    #     for j in theta2:
    #         profile_prob1 = profile_line(map, robot_position[0], robot_position[1], linewidth=2, order=0, reduce_func=None)
    #         profile_prob2 = profile_line(map, robot_position[1], robot_position[2], linewidth=2, order=0, reduce_func=None)
    #         profile_prob = np.concatenate((profile_prob1, profile_prob2))
            
    #         if 0 in profile_prob:
    #             prob = 0
    #         else:
    #             prob = np.min(profile_prob)
    #         configuration_space.append([i, j, prob])