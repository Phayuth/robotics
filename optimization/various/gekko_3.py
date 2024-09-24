from gekko import GEKKO
import numpy as np
import matplotlib.pyplot as plt

# optimal control minimum flight time

m = GEKKO()

nt = 101
tm = np.linspace(0, 1, nt)
m.time = tm

x1 = m.Var(value=-2.5, lb=-100, ub=100)
x2 = m.Var(value=0, lb=-100, ub=100)
u = m.MV(value=0, lb=-np.pi, ub=np.pi)
u.STATUS = 1
u.DCOST = 0.1

p = np.zeros(nt)
p[-1] = 1.0
final = m.Param(value=p)

tf = m.FV(value=10, lb=0.1, ub=100.0)
tf.STATUS = 1

c = m.if3(x2 - 1, 1, (x2 - 1) ** 2 + 1)  # if x2 <= 1 then 1, if x2 > 1 then (x2-1)**1 + 1
m.Equation(x1.dt() == c * m.cos(u) * tf)
m.Equation(x2.dt() == c * m.sin(u) * tf)

# hard constraints (fix endpoint)
# m.fix_final(x1,3)
# m.fix_final(x2,0)

# soft constraints (objective)
m.Minimize(100 * final * (x1 - 3) ** 2)
m.Minimize(100 * final * (x2 - 0) ** 2)

# minimize final time
# initialize with IPOPT Solver
m.Minimize(tf)
m.options.IMODE = 6
m.options.SOLVER = 3
m.solve()

# find MINLP solution with APOPT Solver
m.options.SOLVER = 1
m.options.TIME_SHIFT = 0
m.solve()

tm = tm * tf.value[0]

plt.figure(figsize=(8, 5))
plt.plot(tm, x1.value, "k-", lw=2, label=r"$x_1$")
plt.plot(tm, x2.value, "b-", lw=2, label=r"$x_2$")
plt.plot(tm, u.value, "r--", lw=2, label=r"$u$")
plt.legend(loc="best")
plt.grid()
plt.xlabel("Time")
plt.ylabel("Value")
plt.show()
