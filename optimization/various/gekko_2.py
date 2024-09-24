from gekko import GEKKO
import numpy as np
import matplotlib.pyplot as plt

# optimal control free terminal time. free terminal time, integral objective and differential equations as constraints

m = GEKKO()

n = 501
tm = np.linspace(0, 1, n)
m.time = tm

# Variables
x1 = m.Var(value=1)
x2 = m.Var(value=2)
u = m.MV(value=-1, fixed_initial=False)
u.STATUS = 1
u.DCOST = 1e-5

p = np.zeros(n)
p[-1] = 1.0
final = m.Param(value=p)

# FV
tf = m.FV(value=10.0, lb=0.0, ub=100.0)
tf.STATUS = 1

# equations
m.Equation(x1.dt() / tf == x2)
m.Equation(x2.dt() / tf == u)

# Final conditions
soft = True
if soft:
    # soft terminal constraint
    m.Minimize(final * 1e5 * (x1 - 3) ** 2)
    # m.Minimize(final*1e5*(x2-2)**2)
else:
    # hard terminal constraint
    x1f = m.Param()
    m.free(x1f)
    m.fix_final(x1f, 3)
    # connect endpoint parameters to x1 and x2
    m.Equations([x1f == x1])

# Objective Function
obj = m.Intermediate(m.integral((1 / 2) * u**2))

m.Minimize(final * obj)

m.options.IMODE = 6
m.options.NODES = 3
m.options.SOLVER = 3
m.options.MAX_ITER = 500
# m.options.MV_TYPE = 0
m.options.DIAGLEVEL = 0
m.solve(disp=True)

# Create a figure
tm = tm * tf.value[0]
plt.figure(figsize=(10, 4))
plt.subplot(2, 2, 1)
# plt.plot([0,1],[1/9,1/9],'r:',label=r'$x<\frac{1}{9}$')
plt.plot(tm, x1.value, "k-", lw=2, label=r"$x1$")
plt.ylabel("x1")
plt.legend(loc="best")
plt.subplot(2, 2, 2)
plt.plot(tm, x2.value, "b--", lw=2, label=r"$x2$")
plt.ylabel("x2")
plt.legend(loc="best")
plt.subplot(2, 2, 3)
plt.plot(tm, u.value, "r--", lw=2, label=r"$u$")
plt.ylabel("control")
plt.legend(loc="best")
plt.xlabel("Time")
plt.subplot(2, 2, 4)
plt.plot(tm, obj.value, "g-", lw=2, label=r"$\frac{1}{2} \int u^2$")
plt.text(0.5, 3.0, "Final Value = " + str(np.round(obj.value[-1], 2)))
plt.ylabel("Objective")
plt.legend(loc="best")
plt.xlabel("Time")
plt.show()
