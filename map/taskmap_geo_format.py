import os
import sys
wd = os.path.abspath(os.getcwd())
sys.path.append(str(wd))

from geometry.geometry_class import ObjRectangle


def task_rectangle_obs_1():
    return [
        ObjRectangle(x=2, y=2, h=2, w=2),
        ObjRectangle(x=-4, y=-0.2, h=6, w=3)]


def task_rectangle_obs_2():
    return [
        ObjRectangle(50, 50, h=3, w=3),
        ObjRectangle(40, 20, h=9, w=1),
        ObjRectangle(70, 90, h=5, w=2)]


def task_rectangle_obs_3():
    return [
        ObjRectangle(x=1, y=1, h=0.5, w=0.5),
        ObjRectangle(x=0, y=1.5, h=1, w=2),
        ObjRectangle(x=-2, y=1, h=1, w=1),
        ObjRectangle(x=-1, y=-1, h=0.8, w=4)]


def task_rectangle_obs_4():
    return [
        ObjRectangle(x=1.5, y=1.56, h=0.2, w=1),
        ObjRectangle(x=1.5, y=0.8, h=0.2, w=1),
        ObjRectangle(x=-4, y=-0.2, h=6, w=3)]


def task_rectangle_obs_5():
    return [
        ObjRectangle(x=1.5, y=2.3, h=0.2, w=1),
        ObjRectangle(x=1.5, y=1.8, h=0.2, w=1)]


def task_rectangle_obs_6():
    return [
        ObjRectangle(x=1.5, y=1.25, h=0.2, w=1, angle=1),
        ObjRectangle(x=1.5, y=0.5, h=0.2, w=1, angle=0.3)]


def task_rectangle_obs_7():
    # for my rrtbase
    return [
        ObjRectangle(x=1, y=1, h=1, w=2),
        ObjRectangle(x=5, y=1, h=1, w=2),
        ObjRectangle(x=1, y=5, h=1, w=1),
        ObjRectangle(x=5, y=5, h=0.8, w=4)]


def task_rectangle_obs_8():
    return [ObjRectangle(x=1.5, y=0.5, h=0.01, w=1)]


def two_near_ee_for_devplanner():
    return [
        ObjRectangle(x=1.5, y=1.25, h=0.2, w=1, p=1),
        ObjRectangle(x=1.5, y=0.5, h=0.2, w=1, p=1),
        ObjRectangle(x=-1, y=0, h=1, w=0.5, p=1),
        ObjRectangle(x=-0.5, y=-0.5, h=0.4, w=3, p=1)]


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    plt.axes().set_aspect('equal')
    plt.axvline(x=0, c="green")
    plt.axhline(y=0, c="green")

    listTask = [task_rectangle_obs_6()]

    for task in listTask:
        for obs in task:
            obs.plot()
    plt.show()