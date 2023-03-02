import os
import sys
wd = os.path.abspath(os.getcwd())
sys.path.append(str(wd))

from collections import deque
import numpy as np
import matplotlib.pyplot as plt

# Parameters
KP = 5.0  # attractive potential gain
ETA = 100.0  # repulsive potential gain
AREA_WIDTH = 30  # potential area width [m]
OSCILLATIONS_DETECTION_LENGTH = 3 # the number of previous positions used to check oscillations


def calc_attractive_potential(x, y, gx, gy):
    return 0.5 * KP * np.hypot(x - gx, y - gy)

def calc_repulsive_potential(x, y, ox, oy, rr):
    # search nearest obstacle
    minid = -1
    dmin = float("inf")
    for i, _ in enumerate(ox):
        d = np.hypot(x - ox[i], y - oy[i])
        if dmin >= d:
            dmin = d
            minid = i

    # calc repulsive potential
    dq = np.hypot(x - ox[minid], y - oy[minid])

    if dq <= rr:
        if dq <= 0.1:
            dq = 0.1
        return 0.5 * ETA * (1.0 / dq - 1.0 / rr) ** 2
    else:
        return 0.0

def calc_potential_field(map, gx, gy, rr, sx, sy): # goal x, goal y, obs x, obs y, robot radius, start x, start y
    
    # Create map of 0 element with the shape like map
    potential_map = np.zeros_like(map)

    # Create list of obstacle coordinate
    ox = [] # Create empty list for obstacle in x
    oy = [] # Create empty list for obstacle in y
    for i in range(len(map[0])):
        for j in range(len(map[1])):
            if map[i,j] == 0: # if any element in map is 0, then it is an obstacle, so add its x,y coordinate to obstacle list
                ox.append(i)
                oy.append(j)

    # calculate max and min potential value for pot map ? maybe ? still guessing
    minx = min(min(ox), sx, gx)# - AREA_WIDTH / 2.0
    miny = min(min(oy), sy, gy)# - AREA_WIDTH / 2.0
    maxx = max(max(ox), sx, gx)# + AREA_WIDTH / 2.0
    maxy = max(max(oy), sy, gy)# + AREA_WIDTH / 2.0

    # for each cell of map, calculate potential force
    for xcell in range(len(potential_map[0])):
        for ycell in range(len(potential_map[1])):
            ug = calc_attractive_potential(xcell, ycell, gx, gy)
            uo = calc_repulsive_potential(xcell, ycell, ox, oy, rr)
            uf = ug + uo
            potential_map[xcell,ycell] = uf
    
    return potential_map, minx, miny, maxx, maxy


def get_motion_model(): # move in 8 directions
    # dx, dy
    motion = [[1, 0],
              [0, 1],
              [-1, 0],
              [0, -1],
              [-1, -1],
              [-1, 1],
              [1, -1],
              [1, 1]]

    return motion


def oscillations_detection(previous_ids, ix, iy): # when the goal is arrived, the path will oscilate back and forth, so we check when the same pose is added again , then we stop it
    previous_ids.append((ix, iy))

    if (len(previous_ids) > OSCILLATIONS_DETECTION_LENGTH):
        previous_ids.popleft()

    # check if contains any duplicates by copying into a set
    previous_ids_set = set()
    for index in previous_ids:
        if index in previous_ids_set:
            return True
        else:
            previous_ids_set.add(index)
    return False


def potential_field_planning(map, sx, sy, gx, gy, rr): # potential field planner

    # calc potential field
    potential_map, minx, miny, maxx, maxy = calc_potential_field(map, gx, gy, rr, sx, sy)

    # search path
    # d = np.hypot(sx - gx, sy - gy)
    # print(d)
    # ix = round((sx - minx) / reso)
    # iy = round((sy - miny) / reso)
    # gix = round((gx - minx) / reso)
    # giy = round((gy - miny) / reso)

    # print(ix)
    # print(iy)
    # print(gix)
    # print(giy)

    # ix = sx # it is already a pixel, I guess ?
    # iy = sy
    # gix = gx
    # giy = gy

    plt.imshow(potential_map)
    plt.plot(sx, sy, "*k")
    plt.plot(gx, gy, "*m")
    plt.show()

    inx_now = sx
    iny_now = sy
    d = np.hypot(inx_now - gx, iny_now - gy)

    rx, ry = [sx], [sy] # result of path planning, ready to append
    motion = get_motion_model()
    previous_ids = deque() # double end queue -> a list that can append or pop at both end

    # gradient desc part? maybe ?
    while d != 0 : # we see that d is hypot of start to goal which give distance, so here if the distance if bigger than 0 mean not arrive at goal yet
        # minp = float("inf")
        # minix, miniy = -1, -1

        
        potv_now = potential_map[inx_now, iny_now]
        # print(potv_now)
        potv_now_up = potential_map[inx_now,iny_now+1]# look up
        potv_now_dn = potential_map[inx_now,iny_now-1]# look down
        potv_now_lf = potential_map[inx_now-1,iny_now]# look left
        potv_now_rt = potential_map[inx_now+1,iny_now]# look right
        potv_now_q1 = potential_map[inx_now+1,iny_now+1]# look quaterant 1
        potv_now_q2 = potential_map[inx_now-1,iny_now+1]# look q2
        potv_now_q3 = potential_map[inx_now-1,iny_now-1]# look q3
        potv_now_q4 = potential_map[inx_now+1,iny_now-1]# look q4

        potv_neigbour = [potv_now_up,potv_now_dn,potv_now_lf,potv_now_rt,potv_now_q1,potv_now_q2,potv_now_q3,potv_now_q4]
        print(potv_neigbour)
        min_index_of_potv_neigbour = np.argmin(potv_neigbour)
        print(min_index_of_potv_neigbour)

        if min_index_of_potv_neigbour == 0:#up
            xnext = 0
            ynext = 1
            print("up")

        if min_index_of_potv_neigbour == 1:#dn
            xnext = 0
            ynext = -1
            print("down")

        if min_index_of_potv_neigbour == 2:#lf
            xnext = -1
            ynext = 0
            print("left")

        if min_index_of_potv_neigbour == 3:#rg
            xnext = 1
            ynext = 0
            print("right")

        if min_index_of_potv_neigbour == 4:#q1
            xnext = 1
            ynext = 1
            print("q1")

        if min_index_of_potv_neigbour == 5:#q2
            xnext = -1
            ynext = 1
            print("q2")

        if min_index_of_potv_neigbour == 6:#q3
            xnext = -1
            ynext = -1
            print("q3")

        if min_index_of_potv_neigbour == 7:#q4
            xnext = 1
            ynext = -1
            print("q4")

        inx_now = inx_now + xnext
        iny_now = iny_now + ynext

        rx.append(inx_now)
        ry.append(iny_now)

        d = np.hypot(inx_now - gx, iny_now - gy)

        # break

        # for i, _ in enumerate(motion): # return "counter number" and "value" inside enumerate
        #     inx = int(ix + motion[i][0])
        #     iny = int(iy + motion[i][1])
        #     if inx >= len(potential_map) or iny >= len(potential_map[0]) or inx < 0 or iny < 0:
        #         p = float("inf")  # outside area
        #         print("outside potential!")
        #     else:
        #         p = potential_map[inx,iny]
        #     if minp > p:
        #         minp = p
        #         minix = inx
        #         miniy = iny
        # ix = minix
        # iy = miniy
        # xp = ix * reso + minx
        # yp = iy * reso + miny
        # d = np.hypot(gx - xp, gy - yp)
        # rx.append(xp)
        # ry.append(yp)

        # if (oscillations_detection(previous_ids, ix, iy)):
        #     print("Oscillation detected at ({},{})!".format(ix, iy))
        #     break

    return potential_map, rx, ry


def draw_heatmap(data):
    data = data.T
    plt.pcolor(data, vmax=100.0, cmap=plt.cm.Blues)

def map_val(x, in_min, in_max, out_min, out_max):
    return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min