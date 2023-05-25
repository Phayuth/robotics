"""
Taskspace map in geometry format made for 6 dof

"""

import os
import sys

wd = os.path.abspath(os.getcwd())
sys.path.append(str(wd))

from collision_check_geometry import collision_class


def two_near_ee():
    rec1 = collision_class.ObjRec(x=1.5, y=1.25, h=0.2, w=1, p=1)
    rec2 = collision_class.ObjRec(x=1.5, y=0.5, h=0.2, w=1, p=1)
    list_obs = [rec1, rec2]
    return list_obs


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    plt.axes().set_aspect('equal')
    plt.axvline(x=0, c="green")
    plt.axhline(y=0, c="green")

    listTask = [two_near_ee()]

    for task in listTask:
        for obs in task:
            obs.plot()
    plt.show()