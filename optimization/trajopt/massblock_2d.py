import matplotlib.pyplot as plt
import numpy as np
from gekko import GEKKO

m = GEKKO()

nt = 501
tm = np.linspace(0, 1, nt)
m.time = tm

# Variables (State)
x1 = m.Var(value=10.0, lb=0, ub=10)
x2 = m.Var(value=10.0, lb=0, ub=10)
x3 = m.Var(value=0.0)
x4 = m.Var(value=0.0)

# FV (Optimize for minimal time tf, and tf is always fixed per simulation)
tf = m.FV(value=1.0, lb=0.1, ub=100.0)
tf.STATUS = 1

# MV (System input, let optimizer find optimal control value for us)
u1 = m.MV(value=0, lb=-2, ub=2)
u2 = m.MV(value=0, lb=-2, ub=2)
u1.STATUS = 1
u2.STATUS = 1

# Fixed desired value for final system state
p = np.zeros(nt)
p[-1] = 1.0
final = m.Param(value=p)
m.Equation(x1 * final <= 0)
m.Equation(x2 * final <= 0)
m.Equation(x3 * final <= 0)
m.Equation(x4 * final <= 0)

# Dynamic constraint
m.Equation(x1.dt() == x3 * tf)
m.Equation(x2.dt() == x4 * tf)
m.Equation(x3.dt() == u1 * tf)
m.Equation(x4.dt() == u2 * tf)

# Collision contraint
circle = [[5, 5, 1], [4, 2, 1], [6, 2, 1], [8, 4, 1]]
for cir in circle:
    m.Equation(m.sqrt((x1 - cir[0])**2 + (x2 - cir[1])**2) >= cir[2])

# Objective to minimize
m.Minimize(tf)
m.options.IMODE = 6
m.solve()

print('Final Time: ' + str(tf.value[0]))
tm = tm * tf.value[0]

fig, axs = plt.subplots(6, 1)
axs[0].plot(tm, x1.value)
axs[1].plot(tm, x2.value)
axs[2].plot(tm, x3.value)
axs[3].plot(tm, x4.value)
axs[4].plot(tm, u1.value)
axs[5].plot(tm, u2.value)
plt.show()

fig, axs = plt.subplots(1, 1)
axs.set_aspect(1)
for cir in circle:
    Drawing_colored_circle = plt.Circle((cir[0], cir[1]), cir[2])
    axs.add_artist(Drawing_colored_circle)
axs.plot(x1.VALUE, x2.VALUE)
axs.grid(True)
plt.show()