import os
import sys

wd = os.path.abspath(os.getcwd())
sys.path.append(str(wd))

import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(suppress=True, linewidth=1000)

Ke = 1.0
fdist = 0.0
m = 1.0

n = 50000
dt = 0.001
q0 = np.array([0.0, 0.0])  # position, velocity
qhist = np.empty((n, 2))
qhist[0] = q0


def system(q, inputforce):
    delq = np.array([q[1], (inputforce - Ke * q[0] - fdist) / m])
    q = q + delq * dt
    return q


for i in range(n - 1):
    force = 2.0
    qhist[i + 1] = system(qhist[i], force)

times = np.linspace(0, dt * n, n)
plt.plot(times, qhist[:, 0])
plt.show()

# ------------------------------------------

Kpf = 3.0
Kvf = 1.5
fd = 5.0


def control(q):
    fe = Ke * q[0]
    ef = fd - fe
    xdot = q[1]
    fcont = m * ((Kpf / Ke) * ef - (Kvf * xdot)) + fd
    return fcont


qcontrolhist = np.empty((n, 2))
qcontrolhist[0] = q0

for i in range(n - 1):
    fcont = control(qcontrolhist[i])
    qcontrolhist[i + 1] = system(qcontrolhist[i], fcont)

plt.plot(times, qcontrolhist[:, 0])
plt.title(f"Control Force of [{fd} N]")
plt.show()
