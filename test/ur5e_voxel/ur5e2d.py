import numpy as np
from skimage.measure import profile_line
from PIL import Image

def T01(th_i = 0.):
    al_i = np.pi/2
    a_i  = 0.
    d_i = 16.25
    th_i = th_i
    T01 = np.array([[np.cos(th_i), -(np.cos(al_i)*np.sin(th_i)), np.sin(al_i)*np.sin(th_i), a_i*np.cos(th_i)],
                    [np.sin(th_i), np.cos(al_i)*np.cos(th_i), -(np.sin(al_i)*np.cos(th_i)), a_i*np.sin(th_i)],
                    [0, np.sin(al_i), np.cos(al_i), d_i],
                    [0, 0, 0, 1]])
    return T01

def T12(th_i = 0.):
    al_i = 0.
    a_i  = -42.50
    d_i = 0
    th_i = th_i
    T12 = np.array([[np.cos(th_i), -(np.cos(al_i)*np.sin(th_i)), np.sin(al_i)*np.sin(th_i), a_i*np.cos(th_i)],
                    [np.sin(th_i), np.cos(al_i)*np.cos(th_i), -(np.sin(al_i)*np.cos(th_i)), a_i*np.sin(th_i)],
                    [0, np.sin(al_i), np.cos(al_i), d_i],
                    [0, 0, 0, 1]])
    return T12

def T23(th_i = 0.):
    al_i = 0.
    a_i  = -39.22
    d_i = 0.
    th_i = th_i
    T23 = np.array([[np.cos(th_i), -(np.cos(al_i)*np.sin(th_i)), np.sin(al_i)*np.sin(th_i), a_i*np.cos(th_i)],
                    [np.sin(th_i), np.cos(al_i)*np.cos(th_i), -(np.sin(al_i)*np.cos(th_i)), a_i*np.sin(th_i)],
                    [0, np.sin(al_i), np.cos(al_i), d_i],
                    [0, 0, 0, 1]])
    return T23

def T34(th_i = 0.):
    al_i = np.pi/2
    a_i  = 0.
    d_i = 13.33
    th_i = th_i
    T34 = np.array([[np.cos(th_i), -(np.cos(al_i)*np.sin(th_i)), np.sin(al_i)*np.sin(th_i), a_i*np.cos(th_i)],
                    [np.sin(th_i), np.cos(al_i)*np.cos(th_i), -(np.sin(al_i)*np.cos(th_i)), a_i*np.sin(th_i)],
                    [0, np.sin(al_i), np.cos(al_i), d_i],
                    [0, 0, 0, 1]])
    return T34

def T45(th_i = 0.):
    al_i = -(np.pi/2)
    a_i  = 0.
    d_i = 9.97
    th_i = th_i
    T45 = np.array([[np.cos(th_i), -(np.cos(al_i)*np.sin(th_i)), np.sin(al_i)*np.sin(th_i), a_i*np.cos(th_i)],
                    [np.sin(th_i), np.cos(al_i)*np.cos(th_i), -(np.sin(al_i)*np.cos(th_i)), a_i*np.sin(th_i)],
                    [0, np.sin(al_i), np.cos(al_i), d_i],
                    [0, 0, 0, 1]])
    return T45

def T56(th_i = np.pi): #0
    al_i = 0.
    a_i  = 0.
    d_i = 9.96
    th_i = th_i
    T56 = np.array([[np.cos(th_i), -(np.cos(al_i)*np.sin(th_i)), np.sin(al_i)*np.sin(th_i), a_i*np.cos(th_i)],
                    [np.sin(th_i), np.cos(al_i)*np.cos(th_i), -(np.sin(al_i)*np.cos(th_i)), a_i*np.sin(th_i)],
                    [0, np.sin(al_i), np.cos(al_i), d_i],
                    [0, 0, 0, 1]])
    return T56

def pmap():
    image = Image.open("./map/mapdata/image/map1.png").convert("RGB")
    map = np.asarray(image)

    r_map = np.zeros([201, 101])
    for i in range(np.shape(map)[0]):
        for j in range(np.shape(map)[1]):
            if map[i][j][0] >= 200 and map[i][j][1] <= 100 and map[i][j][2] <= 100:
                r_map[i][j] = 1
            elif map[i][j][0] <= 100 and map[i][j][1] <= 100 and map[i][j][2] >= 200:
                r_map[i][j] = 1#0.8
            elif map[i][j][0] <= 100 and map[i][j][1] >= 200 and map[i][j][2] <= 100:
                r_map[i][j] = 1#0.3
            # elif map[i][j][0] <= 150 and map[i][j][1] <= 150 and map[i][j][2] <= 150:
            #     r_map[i][j] = 0.8

    return 1 - r_map

class Robot(object):

    def __init__(self, base_position):
        self.base_position = base_position

    def robot_position(self, theta2, theta3):
        position = []
        theta2 = (theta2 * np.pi) / 180
        theta3 = (theta3 * np.pi) / 180

        t01 = T01(0)
        x1 = self.base_position[0] + t01[0,3]
        z1 = self.base_position[1] + t01[2,3]
        position.append([x1, z1])

        t02 = np.dot(t01, T12(theta2))
        x2 = self.base_position[0] + t02[0,3]
        z2 = self.base_position[1] + t02[2,3]
        position.append([x2, z2])

        t03 = np.dot(t02, T23(theta3))
        t04 = np.dot(t03, T34(-np.pi/2))
        t05 = np.dot(t04, T45())

        t06 = np.dot(t05, T56())
        x3 = self.base_position[0] + t06[0,3]
        z3 = self.base_position[1] + t06[2,3]
        position.append([x3, z3])

        return position

    def construct_config_space(self, map, grid = 361):

        configuration_space = []
        theta1, theta2 = np.linspace(-180, 180, grid), np.linspace(-180, 180, grid)

        for i in theta1:
            for j in theta2:

                prob = 0
                robot_position = self.robot_position(i, j)

                if robot_position[1][1] <= self.base_position[1] or robot_position[2][1] <= self.base_position[1]:
                    prob = 0

                else:
                    profile_prob1 = np.ravel(profile_line(map, robot_position[0], robot_position[1], linewidth=8, order=0, reduce_func=None))
                    profile_prob2 = np.ravel(profile_line(map, robot_position[1], robot_position[2], linewidth=7, order=0, reduce_func=None))

                    profile_prob = np.concatenate((profile_prob1, profile_prob2))
                    if 0 in profile_prob:
                        prob = 0
                    else:
                        prob = np.min(profile_prob)

                configuration_space.append([i, j, prob])
            print("done")

        c_map = np.zeros((361,361))
        for i in range(361):
            for j in range(361):
                c_map[i][j] = configuration_space[i*361+j][2]

        return c_map
