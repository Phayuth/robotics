from gekko import GEKKO
import numpy as np
import matplotlib.pyplot as plt


# optimal_control_intergral_obj
m = GEKKO()

nt = 101
m.time = np.linspace(0, 2, nt)

x1 = m.Var(value=1)
x2 = m.Var(value=0)
# u = m.Var(value=0, lb=-1, ub=1)
u = m.MV(value=0, lb=-1, ub=1)
u.STATUS = 1

p = np.zeros(nt)  # mark final time point
p[-1] = 1.0
final = m.Param(value=p)

# Equations
m.Equation(x1.dt() == u)
m.Equation(x2.dt() == (1 / 2) * x1**2)
m.Obj(x2 * final)  # Objective function

m.options.IMODE = 6  # optimal control mode
m.solve(disp=True)  # solve

plt.figure(1)  # plot results
plt.plot(m.time, x1.value, "k-", label=r"$x_1$")
plt.plot(m.time, x2.value, "b-", label=r"$x_2$")
plt.plot(m.time, u.value, "r--", label=r"$u$")
plt.legend(loc="best")
plt.xlabel("Time")
plt.ylabel("Value")
plt.show()
