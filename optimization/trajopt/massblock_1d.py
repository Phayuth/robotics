import matplotlib.pyplot as plt
import numpy as np
from gekko import GEKKO

v_max = 30  # maximum velocity
v_min = -4  # minimum velocity
av_max = 2  # maximum acceleration
av_min = -2 # minimum acceleration (ie. maximum deceleration)
x_s = 0     # starting position
x_f = 10    # ending position
mass = 1    # kg

m = GEKKO()

nt = 501
times = np.linspace(0, 1, nt)
m.time = times

x = m.Var(value=x_s)
v = m.Var(value=0.0, lb=v_min, ub=v_max)

av = m.MV(value=0, lb=av_min, ub=av_max)
av.STATUS = 1

tf = m.FV(value=1.0, lb=0.1, ub=100.0)
tf.STATUS = 1

m.Equation(x.dt() == v * tf)
m.Equation(v.dt() == (av/mass) * tf)

# path constraint
m.Equation(x >= 0)

# boundary constraints
m.fix(x, pos=len(m.time) - 1, val=x_f)  # vehicle must arrive at x=10
m.fix(v, pos=len(m.time) - 1, val=0.0)  # vehicle must come to a full stop

# objective - minimise the travel time
m.Obj(tf)

# solve
m.options.IMODE = 6
m.solve(disp=True)

# plot solution
tm = times * tf
plt.figure(1)
plt.title('Final Time: ' + str(tf.value[0]))
plt.plot(tm, x.value, 'k-', label=r'$x$')
plt.plot(tm, v.value, 'g-', label=r'$v$')
plt.plot(tm, av.value, 'r--', label=r'$a_v$')
plt.legend(loc='best')
plt.xlabel('Time')
plt.show()