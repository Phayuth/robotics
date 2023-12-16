import multiprocessing
import time


def funx(y, x):
    print(f"startx = {y}, input = {x}")
    time.sleep(y)
    print("donex")


def funy(y, x):
    print(f"starty = {y}, input = {x}")
    time.sleep(y)
    print("doney")


x = multiprocessing.Process(target=funx, args=[2, "Hi"])
x.start()

y = multiprocessing.Process(target=funy, args=[2, "Hello"])
y.start()

x.join()
y.join()