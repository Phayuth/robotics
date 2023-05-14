import matplotlib.pyplot as plt
import numpy as np

q0=-5; qf = 80; tf = 4

min_acc = 4*abs(qf-q0)/tf**2
print("acc needs to be bigger than ", min_acc)

mode_aorv = 0

if mode_aorv == 0:
    acc = 30
    tb = tf/2 -np.sqrt( acc**2 * tf**2 -4*acc*np.abs(qf-q0)) /2/acc
    if (qf-q0)>=0:
        vel = acc * tb
    else:
        acc = acc * -1
        vel = acc * tb
else:
    vel = 25
    tb = (vel*tf - abs(qf-q0))/vel
    if (qf-q0)>=0:
        acc = vel / tb
    else:
        vel = vel * -1
        acc = vel/tb

# Acceleration Region  # 1st Region
t1 = np.arange(0, tb, step=0.01)
q1 = q0 + vel/(2*tb) * (t1**2)
dq1 = acc * t1
ddq1 = acc*np.ones(np.size(t1))

# Constant Velocity Region # 2nd Region
t2   = np.arange(tb, tf-tb, step=0.01)
q2  = (qf+q0-vel*tf)/2 + vel* t2
dq2 = vel*np.ones(np.size(t2))
ddq2 = 0*np.ones(np.size(t2))

# Decceleration Region # 3rd Region
t3 = np.arange(tf-tb, tf+0.01, step=0.01)
q3 = qf - acc/2 * tf**2 + acc*tf*t3 - acc/2*t3**2
dq3 = acc * tf - acc*t3
ddq3 = -acc * np.ones(np.size(t3))

# Total Region
t = np.concatenate((t1,t2,t3))
q = np.concatenate((q1,q2,q3))
dq = np.concatenate((dq1,dq2,dq3))
ddq = np.concatenate((ddq1,ddq2,ddq3))

# Plotting Graph
fig, (ax1, ax2, ax3) = plt.subplots(3)

ax1.set(xlabel = "time is second", ylabel = "joint pose in deg")
ax1.plot(t, q)

ax2.set(xlabel = "time is second", ylabel = "joint vel in deg/s")
ax2.plot(t, dq)

ax3.set(xlabel = "time is second", ylabel = "joint acc in deg/sec^2")
ax3.plot(t, ddq)

plt.show()
