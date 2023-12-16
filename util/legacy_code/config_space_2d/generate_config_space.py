import numpy as np
from skimage.measure import profile_line
    
def construct_config_space_2d(robot, map, grid = 361):

    configurationSpace = []
    theta1, theta2 = np.linspace(0, 360, grid), np.linspace(0, 360, grid)

    for i in theta1:
        for j in theta2:

            robotPosition = robot.robot_position(i, j)

            prob = 0
            profileProb1 = profile_line(map, robotPosition[0], robotPosition[1], linewidth=2, order=0, reduce_func=None)
            profileProb2 = profile_line(map, robotPosition[1], robotPosition[2], linewidth=2, order=0, reduce_func=None)
            profileProb = np.concatenate((profileProb1, profileProb2))
            
            if 0 in profileProb:
                prob = 0
            else:
                prob = np.min(profileProb)

            configurationSpace.append([i, j, prob])

            print(f"At theta 1: {i} | At theta 2: {j}")

    cMap = np.zeros((361,361))
    for i in range(361):
        for j in range(361):
            cMap[i][j] = configurationSpace[i*361+j][2]

    return cMap

if __name__=="__main__":
    import os
    import sys
    wd = os.path.abspath(os.getcwd())
    sys.path.append(str(wd))

    from robot.planar_rr import PlanarRRVoxel
    from map import taskmap_img_format
    from generate_config_space import construct_config_space_2d
    import matplotlib.pyplot as plt

    # Create map
    map = taskmap_img_format.map_2d_1()

    # Create robot
    basePosition = [15, 15]
    linkLenths = [5, 5]
    robot = PlanarRRVoxel(basePosition, linkLenths)

    # Create Configuration space
    configuration = construct_config_space_2d(robot, map)

    # Plot robot and obstacle in task space
    plt.figure(figsize=(10,10))
    plt.axes().set_aspect('equal')
    r1 = robot.robot_position(90,0)
    plt.plot([basePosition[0], r1[0][0]],[basePosition[1], r1[0][1]] , "b", linewidth=8)
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