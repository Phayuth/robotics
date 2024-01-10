# Notebook 7.1 - Introduction to Trajectory Optimisation in Python
# https://transport-systems.imperial.ac.uk/tf/60008_21/n7_1_introduction_to_trajectory_optimisation.html
# https://www.youtube.com/playlist?list=PLLBUgWXdTBDjxcpH9hRuq-bsm_ti2UvoB


from gekko import GEKKO
import numpy as np
import matplotlib.pyplot as plt

v_max = 30  # maximum velocity
v_min = -4  # minimum velocity
av_max = 2  # maximum acceleration
av_min = -2  # minimum acceleration (ie. maximum deceleration)
x_s = 0  # starting position
x_f = 10  # ending position
v_bound = 0  # velocity in the start and end of the journey

# m = GEKKO()

# nt = 501
# m.time = np.linspace(0, 1, nt)

# x = m.Var(value=0.0)
# v = m.Var(value=0.0, lb=v_min, ub=v_max)

# av = m.MV(value=0, lb=av_min, ub=av_max)
# av.STATUS = 1

# tf = m.FV(value=1.0, lb=0.1, ub=100.0)
# tf.STATUS = 1

# m.Equation(x.dt() == v * tf)
# m.Equation(v.dt() == av * tf)

# # path constraint
# m.Equation(x >= 0)

# # boundary constraints
# m.fix(x, pos=len(m.time) - 1, val=10.0)  # vehicle must arrive at x=10
# m.fix(v, pos=len(m.time) - 1, val=0.0)  # vehicle must come to a full stop

# # objective - minimise the travel time
# m.Obj(tf)

# # solve
# m.options.IMODE = 6
# m.solve(disp=False)

# # print final travel time
# print('Final Time: ' + str(tf.value[0]))

# # plot solution
# tm = np.linspace(0, tf.value[0], nt)
# plt.figure(1)
# plt.plot(tm, x.value, 'k-', label=r'$x$')
# plt.plot(tm, v.value, 'g-', label=r'$v$')
# plt.plot(tm, av.value, 'r--', label=r'$a_v$')
# plt.legend(loc='best')
# plt.xlabel('Time')
# plt.show()






# 2. Mathematical model - 2D Vehicle Movement

phi_max = 0.7

# initialize GEKKO
m = GEKKO(remote=True)
nt = 501
m.time = np.linspace(0, 1, nt)

# Variables
x = m.Var(value=0)
y = m.Var(value=0)
v = m.Var(value=0, lb=v_min, ub=v_max)
th = m.Var(value=0, lb=-np.pi / 2, ub=np.pi / 2)  # bound heading to -pi/2 and pi/2

# optimize final time
tf = m.FV(value=1.0, lb=0.1, ub=100.0)
tf.STATUS = 1

# control changes every time period
av = m.MV(value=0, lb=-2, ub=2)
av.STATUS = 1
phi = m.MV(value=0, lb=-phi_max, ub=phi_max)
phi.STATUS = 1

# define the ODEs that describe the movement of the vehicle
m.Equation(x.dt() == v * m.cos(th) * tf)
m.Equation(y.dt() == v * m.sin(th) * tf)
m.Equation(v.dt() == av * tf)
m.Equation(th.dt() == phi * tf)

# define path constraints
m.Equation(x >= 0)
m.Equation(y >= 0)

# define voundary constraints
m.fix(x, pos=len(m.time) - 1, val=7.0)
m.fix(y, pos=len(m.time) - 1, val=10.0)
m.fix(v, pos=len(m.time) - 1, val=0)
m.fix(th, pos=len(m.time) - 1, val=0.1)

# define voundary constraints
m.fix(x, pos=200 - 1, val=2.0)
m.fix(y, pos=200 - 1, val=4.0)

# Objective function
m.Obj(tf)

# Solve
m.options.IMODE = 6
m.solve(disp=True)  # set to True to view solver logs

#Presentation of results
print('Final Time: ' + str(tf.value[0]))

#Plot solution
tm = np.linspace(0, tf.value[0], nt)
plt.figure(1)
plt.plot(tm, x.value, 'k-', label=r'$x$')
plt.plot(tm, y.value, 'b-', label=r'$y$')
plt.plot(tm, v.value, 'g-', label=r'$v$')
plt.plot(tm, th.value, 'm-', label=r'$\theta$')
plt.plot(tm, av.value, 'r--', label=r'$a_v$')
plt.plot(tm, phi.value, 'y--', label=r'$\phi$')
plt.legend(loc='best')
plt.xlabel('Time')
plt.show()

plt.figure(1)
plt.plot(x.value, y.value, 'k-', label=r'vehicle')
plt.legend(loc='best')
plt.xlabel('x position')
plt.ylabel('y position')
plt.show()
