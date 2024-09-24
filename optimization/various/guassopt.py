import numpy as np
import matplotlib.pyplot as plt


def f1(x):
    a = 10
    b = 0
    c = 1
    y = a * np.exp(-((x - b) ** 2) / (2 * c**2))
    return y


def f2(x):
    a = 5
    b = 3
    c = 4
    y = a * np.exp(-((x - b) ** 2) / (2 * c**2))
    return y


x = np.linspace(-10, 10, 1000)
y1 = f1(x)
y2 = f2(x)

plt.plot(x, y1)
plt.plot(x, y2)
plt.show()
