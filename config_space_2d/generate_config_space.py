import numpy as np
from skimage.measure import profile_line
    
def construct_config_space_2d(robot, map, grid = 361):

    configuration_space = []
    theta1, theta2 = np.linspace(0, 360, grid), np.linspace(0, 360, grid)

    for i in theta1:
        for j in theta2:

            robot_position = robot.robot_position(i, j)

            prob = 0
            profile_prob1 = profile_line(map, robot_position[0], robot_position[1], linewidth=2, order=0, reduce_func=None)
            profile_prob2 = profile_line(map, robot_position[1], robot_position[2], linewidth=2, order=0, reduce_func=None)
            profile_prob = np.concatenate((profile_prob1, profile_prob2))
            
            if 0 in profile_prob:
                prob = 0
            else:
                prob = np.min(profile_prob)

            configuration_space.append([i, j, prob])

            print(f"At theta 1: {i} | At theta 2: {j}")

    c_map = np.zeros((361,361))
    for i in range(361):
        for j in range(361):
            c_map[i][j] = configuration_space[i*361+j][2]

    return c_map
