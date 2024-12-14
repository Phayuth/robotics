import os
import sys

sys.path.append(str(os.path.abspath(os.getcwd())))

import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(suppress=True, linewidth=1000)

Ke = 250.0
m = 1.0

Kpf = 1.0
Kvf = 28
fd = 30.0


def system(q, inputforce, dt=0.001, fdist = 0.0):
    delq = np.array([q[1], (inputforce - Ke * q[0] - fdist) / m])
    q = q + delq * dt
    return q


def control(q):
    fe = Ke * q[0]
    ef = fd - fe
    xdot = q[1]
    fcont = m * ((Kpf / Ke) * ef - (Kvf * xdot)) + fd
    return fcont


n = 20000
dt = 0.001

q0 = np.array([0.0, 0.0])  # position, velocity
times = np.linspace(0, dt * n, n)
qhist = np.empty((n, 2))
qhist[0] = q0
fhist = [np.nan]

for i in range(n - 1):
    fcont = control(qhist[i])

    if i < 10000:
        fdist = 0.0
    else:
        fdist = 50.0
    qhist[i + 1] = system(qhist[i], fcont, fdist=fdist)
    fhist.append(fcont)

fig, ax = plt.subplots()
ax.plot(times, qhist[:, 0], label="Position History")
ax.set_xlabel("times [s]")
ax.set_ylabel("position [m]")
ax.set_title("Position History")
ax.grid(True)
ax.legend()


fig2, ax2 = plt.subplots()
ax2.plot(times, fhist)
ax2.plot(times, fd * np.ones(times.shape), label="Desired Force")
ax2.set_xlabel("times [s]")
ax2.set_ylabel("force input [N]")
ax2.set_title("Force History")
ax2.grid(True)
ax2.legend()
plt.show()
