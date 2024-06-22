import threading
import time


def funx(y, x):
    print(f"startx = {y}, input = {x}")
    time.sleep(y)
    print("donex")


def funy(y, x):
    print(f"starty = {y}, input = {x}")
    time.sleep(y)
    print("doney")


x = threading.Thread(target=funx, args=[2, "Hi"])

y = threading.Thread(target=funy, args=[2, "Hello"])

x.start()
y.start()

x.join()
y.join()