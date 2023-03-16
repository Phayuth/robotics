import os
import sys
wd = os.path.abspath(os.getcwd())
sys.path.append(str(wd))

import matplotlib.pyplot as plt
from collision_check_geometry import collision_class

def task_rectangle_obs():
    rec1 = collision_class.sqr_rec_2d_obj(x=2,y=2,h=2,w=2)
    rec2 = collision_class.sqr_rec_2d_obj(x=-4, y=-0.2,h=6, w=3)
    list_obs = [rec1,rec2]
    return list_obs

if __name__ == "__main__":

    plt.axes().set_aspect('equal')
    plt.axvline(x=0, c="green")
    plt.axhline(y=0, c="green")

    list_obs = task_rectangle_obs()
    for obs in list_obs:
        obs.plot()
    plt.show()