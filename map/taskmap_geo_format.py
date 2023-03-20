import os
import sys
wd = os.path.abspath(os.getcwd())
sys.path.append(str(wd))

import matplotlib.pyplot as plt
from collision_check_geometry import collision_class

def task_rectangle_obs_1():
    rec1 = collision_class.sqr_rec_2d_obj(x=2,y=2,h=2,w=2)
    rec2 = collision_class.sqr_rec_2d_obj(x=-4,y=-0.2,h=6,w=3)
    list_obs = [rec1,rec2]
    return list_obs

def task_rectangle_obs_2():
    rec1 = collision_class.sqr_rec_2d_obj(50,50,h=3,w=3)
    rec2 = collision_class.sqr_rec_2d_obj(40,20,h=9,w=1)
    rec3 = collision_class.sqr_rec_2d_obj(70,90,h=5,w=2)
    list_obs = [rec1, rec2, rec3]
    return list_obs

def task_rectangle_obs_3():
    rec1 = collision_class.sqr_rec_2d_obj(x=1,y=1,h=0.5,w=0.5)
    rec2 = collision_class.sqr_rec_2d_obj(x=0,y=1.5,h=1,w=2)
    rec3 = collision_class.sqr_rec_2d_obj(x=-2,y=1,h=1,w=1)
    rec4 = collision_class.sqr_rec_2d_obj(x=-1,y=-1,h=0.8,w=4)
    list_obs = [rec1,rec2,rec3,rec4]
    return list_obs

if __name__ == "__main__":

    plt.axes().set_aspect('equal')
    plt.axvline(x=0, c="green")
    plt.axhline(y=0, c="green")

    list_obs = task_rectangle_obs_3()
    for obs in list_obs:
        obs.plot()
    plt.show()