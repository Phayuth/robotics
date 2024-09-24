from gekko import GEKKO
import matplotlib.pyplot as plt
import numpy as np


xd = 3
yd = 1
xinit = 0.5
yinit = 0.5

m = GEKKO()
x = m.Var(value=xinit)
y = m.Var(value=yinit)

m.Obj((xd - x) ** 2 + (yd - y) ** 2) # minimize square cost
# m.Obj(-((xd - x) ** 2 + (yd - y) ** 2)) # maximize square cost

m.Equations(
    [
        2 * x + y - 4 <= 0,
        -0.5 * x + y - 2 <= 0,
        -x + 0 * y <= 0,
        0 * x - y <= 0,
    ]
)  # equations
m.solve(disp=False)
print([x.value, y.value])


xi = np.linspace(-1, 4, 100)


def ff1(x):
    return -2 * x + 4


def ff2(x):
    return 0.5 * x + 2


plt.plot(xi, ff1(xi))
plt.plot(xi, ff2(xi))
plt.plot(xd, yd, "r*", label="desired pose")
plt.plot(xinit, yinit, "b^", label="initial guess pose")
plt.plot(x.value[0], y.value[0], "g+", label="optimizer pose")
plt.axhline()
plt.axvline()
plt.xlim(-1, 4)
plt.ylim(-1, 4)
plt.legend()
plt.show()
