import matplotlib.pyplot as plt
import numpy as np


x = np.linspace(-1, 4, 100)
y = np.linspace(-1, 4, 100)
X, Y = np.meshgrid(x, y)

def fx1(x, y):
    return 2 * x + y - 4 <= 0


def fx2(x, y):
    return -0.5 * x + y - 2 <= 0


def fx3(x, y):
    return -x + 0 * y <= 0


def fx4(x, y):
    return 0 * x - y <= 0


def ff1(x):
    return -2 * x + 4


def ff2(x):
    return 0.5 * x + 2


FX1 = fx1(X, Y)
FX2 = fx2(X, Y)
FX3 = fx3(X, Y)
FX4 = fx4(X, Y)

plt.plot(X[FX1], Y[FX1], "b*")
plt.plot(X[FX2], Y[FX2], "r.")
plt.plot(X[FX3], Y[FX3], "g+")
plt.plot(X[FX4], Y[FX4], "k,")
plt.plot(x, ff1(x))
plt.plot(x, ff2(x))
plt.axhline()
plt.axvline()
plt.xlim(-1, 4)
plt.ylim(-1, 4)
plt.show()

merge = FX1 & FX2 & FX3 & FX4

plt.plot(X[merge], Y[merge], "b*")
plt.plot(X[merge], Y[merge], "r.")
plt.plot(X[merge], Y[merge], "g+")
plt.plot(X[merge], Y[merge], "k,")
plt.plot(x, ff1(x))
plt.plot(x, ff2(x))
plt.axhline()
plt.axvline()
plt.xlim(-1, 4)
plt.ylim(-1, 4)
plt.show()
