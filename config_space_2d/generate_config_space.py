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

if __name__=="__main__":
    import os
    import sys
    wd = os.path.abspath(os.getcwd())
    sys.path.append(str(wd))

    from robot_used.plannar_rr import RobotArm2D
    from map import taskmap_img_format
    from generate_config_space import construct_config_space_2d
    import matplotlib.pyplot as plt

    # Create map
    map = taskmap_img_format.map_2d_1()

    # Create robot
    base_position = [15, 15]
    link_lenths = [5, 5]
    robot = RobotArm2D.Robot(base_position, link_lenths)

    # Create Configuration space
    configuration = construct_config_space_2d(robot, map)

    # Plot robot and obstacle in task space
    plt.figure(figsize=(10,10))
    plt.axes().set_aspect('equal')
    r1 = robot.robot_position(90,0)
    plt.plot([robot.base_position[0], r1[0][0]],[robot.base_position[1], r1[0][1]] , "b", linewidth=8)
    plt.plot([r1[0][0], r1[1][0]],[r1[0][1], r1[1][1]] , "b", linewidth=8)
    plt.plot([r1[1][0], r1[2][0]],[r1[1][1], r1[2][1]] , "r", linewidth=8)
    plt.imshow(np.transpose(map),cmap = "gray", interpolation = 'nearest')
    plt.gca().invert_yaxis()
    plt.show()

    # Plot config space
    plt.figure(figsize=(10,10))
    plt.axes().set_aspect('equal')
    plt.imshow(np.transpose(configuration),cmap = "gray", interpolation = 'nearest')
    plt.show()