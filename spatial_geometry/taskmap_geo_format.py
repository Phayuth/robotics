import os
import sys

sys.path.append(str(os.path.abspath(os.getcwd())))

from spatial_geometry.spatial_shape import ShapeRectangle


class NonMobileTaskMap:

    def task_rectangle_obs_1():
        return [
            ShapeRectangle(x=2, y=2, h=2, w=2),
            ShapeRectangle(x=-4, y=-0.2, h=6, w=3),
        ]

    def task_rectangle_obs_2():
        return [
            ShapeRectangle(50, 50, h=3, w=3),
            ShapeRectangle(40, 20, h=9, w=1),
            ShapeRectangle(70, 90, h=5, w=2),
        ]

    def task_rectangle_obs_3():
        return [
            ShapeRectangle(x=1, y=1, h=0.5, w=0.5),
            ShapeRectangle(x=0, y=1.5, h=1, w=2),
            ShapeRectangle(x=-2, y=1, h=1, w=1),
            ShapeRectangle(x=-1, y=-1, h=0.8, w=4),
        ]

    def task_rectangle_obs_4():
        return [
            ShapeRectangle(x=1.5, y=1.56, h=0.2, w=1),
            ShapeRectangle(x=1.5, y=0.8, h=0.2, w=1),
            ShapeRectangle(x=-4, y=-0.2, h=6, w=3),
        ]

    def task_rectangle_obs_5():
        return [
            ShapeRectangle(x=1.5, y=2.3, h=0.2, w=1),
            ShapeRectangle(x=1.5, y=1.8, h=0.2, w=1),
        ]

    def task_rectangle_obs_6():
        return [
            ShapeRectangle(x=1.5, y=1.25, h=0.2, w=1, angle=1),
            ShapeRectangle(x=1.5, y=0.5, h=0.2, w=1, angle=0.3),
        ]

    def task_rectangle_obs_7():  # for my rrtbase
        return [
            ShapeRectangle(x=1, y=1, h=1, w=2),
            ShapeRectangle(x=5, y=1, h=1, w=2),
            ShapeRectangle(x=1, y=5, h=1, w=1),
            ShapeRectangle(x=5, y=5, h=0.8, w=4),
        ]

    def task_rectangle_obs_8():
        return [
            ShapeRectangle(x=1.5, y=0.5, h=0.01, w=1),
        ]

    def two_near_ee_for_devplanner():
        return [
            ShapeRectangle(x=1.5, y=1.25, h=0.2, w=1, p=1),
            ShapeRectangle(x=1.5, y=0.5, h=0.2, w=1, p=1),
            ShapeRectangle(x=-1, y=0, h=1, w=0.5, p=1),
            ShapeRectangle(x=-0.5, y=-0.5, h=0.4, w=3, p=1),
        ]

    def thesis_exp():
        return [
            ShapeRectangle(x=-2.75, y=1, h=2, w=1),
            ShapeRectangle(x=1.5, y=2, h=2, w=1),
        ]

    def ijcas_paper():
        return [
            ShapeRectangle(x=-2.75, y=1, h=2, w=1),
            ShapeRectangle(x=1.5, y=2, h=2, w=1),
            ShapeRectangle(x=-0.75, y=-2.0, h=0.75, w=4.0),
        ]

    def paper_torus_exp():
        return [
            ShapeRectangle(x=2, y=2, h=2, w=2),
            ShapeRectangle(x=-4, y=2, h=2, w=2),
            ShapeRectangle(x=2, y=-4, h=2, w=2),
            ShapeRectangle(x=-4, y=-4, h=2, w=2),
        ]


class MobileTaskMap:

    def no_collision_map():
        return [
            ShapeRectangle(x=-10, y=-10, h=0.01, w=0.01, p=1),
            ShapeRectangle(x=10, y=-10, h=0.01, w=0.01, p=1),
            ShapeRectangle(x=10, y=10, h=0.01, w=0.01, p=1),
            ShapeRectangle(x=-10, y=10, h=0.01, w=0.01, p=1),
        ]

    def warehouse_map():
        wT = 0.5  # m wallThinkness
        rW = 2  # m shelf width
        rH = 6  # m shelf hight
        tW = 2  # m table
        tH = 1  # m table
        return [
            ShapeRectangle(x=0, y=0, h=8, w=wT, p=1),
            ShapeRectangle(x=0, y=0, h=wT, w=8, p=1),
            ShapeRectangle(x=0, y=5, h=wT, w=8, p=1),
            ShapeRectangle(x=0, y=17.5, h=wT, w=8, p=1),
            ShapeRectangle(x=0, y=10, h=8, w=wT, p=1),
            ShapeRectangle(x=10, y=17.5, h=wT, w=20, p=1),
            ShapeRectangle(x=10, y=5, h=wT, w=8, p=1),
            ShapeRectangle(x=10, y=0, h=wT, w=8, p=1),
            ShapeRectangle(x=18, y=0, h=8, w=wT, p=1),
            ShapeRectangle(x=18.5, y=7.5, h=wT, w=2, p=1),
            ShapeRectangle(x=25.0, y=7.5, h=wT, w=5, p=1),
            ShapeRectangle(x=29.5, y=7.5, h=10, w=wT, p=1),
            ShapeRectangle(x=20, y=12.5, h=wT, w=10, p=1),
            ShapeRectangle(x=20, y=15, h=2.5, w=wT, p=1),
            ShapeRectangle(x=8, y=0, h=2.5, w=wT, p=1),
            ShapeRectangle(x=10, y=0, h=2.5, w=wT, p=1),
            ShapeRectangle(x=2, y=10, h=rH, w=rW, p=1),  # shelf
            ShapeRectangle(x=6, y=10, h=rH, w=rW, p=1),  # shelf
            ShapeRectangle(x=13, y=7.5, h=tH, w=tW, p=1),  # table
            ShapeRectangle(x=13, y=10, h=tH, w=tW, p=1),  # table
            ShapeRectangle(x=13, y=12.5, h=tH, w=tW, p=1),  # table
            ShapeRectangle(x=13, y=15, h=tH, w=tW, p=1),  # table
            ShapeRectangle(x=20, y=0, h=5, w=10, p=1),  # car
        ]  # start robot at 15, 2.5 as charging station

    def warehouse_map_name():
        plt.text(25, 2.5, "Delivery Vehicle", bbox=dict(facecolor="yellow", alpha=0.5))
        plt.text(15, 2.5, "Charging Station", bbox=dict(facecolor="yellow", alpha=0.5))
        plt.text(1, 1, "Control Office", bbox=dict(facecolor="yellow", alpha=0.5))
        plt.text(25, 14, "Head Office", bbox=dict(facecolor="yellow", alpha=0.5))
        plt.text(2, 10, "Large\nStorage", bbox=dict(facecolor="yellow", alpha=0.5))
        plt.text(6, 10, "Large\nStorage", bbox=dict(facecolor="yellow", alpha=0.5))
        plt.text(13, 16.5, "Small\nStorage", bbox=dict(facecolor="yellow", alpha=0.5))


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    plt.axes().set_aspect("equal")
    # plt.grid(True)
    plt.axvline(x=0, c="green")
    plt.axhline(y=0, c="green")
    plt.xlim(-5, 5)
    plt.ylim(-2, 5)

    def plot_warehouse():
        listTask = [MobileTaskMap.warehouse_map()]

        for task in listTask:
            for obs in task:
                obs.plot()
        plt.show()

    def plot_rect():
        listTask = [NonMobileTaskMap.thesis_exp()]
        listTask = [NonMobileTaskMap.ijcas_paper()]

        for task in listTask:
            for obs in task:
                obs.plot()
        plt.show()

    plot_rect()
