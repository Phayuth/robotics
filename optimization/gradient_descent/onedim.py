# https://d2l.ai/chapter_optimization/gd.html
import numpy as np
import matplotlib.pyplot as plt


class Parabola:

    def __init__(self, a, b, c):
        self.a = a
        self.b = b
        self.c = c

    def f(self, x):
        return self.a * x**2 + self.b * x + self.c

    def f_grad_analytical(self, x):
        return 2 * self.a * x + self.b


# gradient descent update | walk down the hill (negative gradient)
xinit = 4
xopt = [xinit]
p = Parabola(1, -4, 5)
eta = 0.1
for i in range(100):
    xinit -= eta * p.f_grad_analytical(xinit)
    xopt.append(xinit)
    print(f"iter {i}, x = {xinit}, y = {p.f(xinit)}")

# plot
x = np.arange(-5, 5, 0.1)
fig, ax = plt.subplots()
ax.plot(x, p.f(x))
ax.plot(xopt, [p.f(x) for x in xopt], "-o", markersize=3, color="r")
ax.set_xlabel("x")
ax.set_ylabel("f(x)")
ax.grid(True)
plt.show()
