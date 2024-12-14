import os
import sys

sys.path.append(str(os.path.abspath(os.getcwd())))

import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(suppress=True, linewidth=1000)

# ------------------------------------------

m = 1.0  # actual system mass


def system(q, forceinput, fe=0.0, dt=0.001):
    # fe external / tip force
    delq = np.array([q[1], (forceinput - fe) / m])
    q = q + delq * dt
    return q


def system_zero_at_xd(q, forceinput, xd=3.0, dt=0.001):
    if q[0] > xd:
        # delq = np.array([0, 0])
        fe = forceinput
        delq = np.array([q[1], (forceinput - fe) / m])
    else:
        fe = 0
        delq = np.array([q[1], (forceinput - fe) / m])
    q = q + delq * dt
    return q


def system_reverse_sign(q, forceinput, fe=0.0, dt=0.001):
    delq = np.array([q[1], (forceinput + fe) / m])
    q = q + delq * dt
    return q


def line_traj(t):
    slope = 2.5
    intercept = 0.2
    xd = slope * t + intercept
    xd_dot = slope
    xd_ddot = 0
    return xd, xd_dot, xd_ddot


def const_acc_traj(t):
    acc = 0.01  # m/s2
    xd_init = 0.2
    xd_dot_init = 0.0

    xd = 0.5 * acc * t**2 + xd_dot_init * t + xd_init
    xd_dot = acc * t + xd_dot_init
    xd_ddot = acc
    return xd, xd_dot, xd_ddot


def const_pose_traj(t):
    xd = 3
    xd_dot = 0
    xd_ddot = 0
    return xd, xd_dot, xd_ddot


def control_book(q, qref, Md=1.0, Kd=1.0, Bd=1.0, fe=0.0):
    e = qref[0] - q[0]
    e_dot = qref[1] - q[1]
    fcont = m * (qref[2] - (Bd * e_dot + Kd * e - fe) / Md) + fe
    return fcont, e


def control_wiki(q, qref, Bd=1.0, Kd=1.0):
    e = qref[0] - q[0]
    e_dot = qref[1] - q[1]
    fcont = m * qref[2] + Bd * e_dot + Kd * e
    return fcont, e


def control_deluca(q, qref, Md=1.0, Kd=1.0, Bd=1.0, fe=0.0):
    e = qref[0] - q[0]
    e_dot = qref[1] - q[1]
    fcont = m * (qref[2] + (Bd * e_dot + Kd * e + fe) / Md) - fe
    return fcont, e


n = 50_000
dt = 0.001
q0 = np.array([0.0, 0.0])  # position, velocity
times = np.linspace(0, dt * n, n)
qhist = np.empty((n, 2))
qhist[0] = q0
qrhist = np.empty((n, 3))
qrhist[0] = np.array([0, 0, 0])

errorhist = [0]
fhist = [0]

for i in range(n - 1):
    # traj
    # qref = const_acc_traj(i * dt)
    qref = const_pose_traj(i * dt)

    # control
    # fcont, e = control_book(qhist[i], qref)
    # fcont, e = control_wiki(qhist[i], qref)
    fcont, e = control_deluca(qhist[i], qref, fe=-1.0, Bd=3)

    # intergrate
    # qhist[i + 1] = system(qhist[i], fcont)
    # qhist[i + 1] = system_zero_at_xd(qhist[i], fcont)
    qhist[i + 1] = system_reverse_sign(qhist[i], fcont, fe=-1.0)

    # save history
    qrhist[i + 1] = np.array(qref)
    errorhist.append(e)
    fhist.append(fcont)


fig, ax = plt.subplots()
ax.plot(times, qhist[:, 0], label="actual position [m]")
ax.plot(times, qrhist[:, 0], label="reference position [m]")
ax.set_title("Tracking Control")
ax.set_xlabel("times [s]")
ax.set_ylabel("position [m]")
ax.grid(True)
ax.legend()

fig1, ax1 = plt.subplots()
ax1.plot(times, errorhist)
ax1.grid(True)
ax1.set_xlabel("times [s]")
ax1.set_ylabel("error in position [m]")
ax1.set_title("Error overtime")

fig2, ax2 = plt.subplots()
ax2.plot(times, fhist)
ax2.set_xlabel("times [s]")
ax2.set_ylabel("force input [N]")
ax2.set_title("Force History")
ax2.grid(True)
plt.show()
