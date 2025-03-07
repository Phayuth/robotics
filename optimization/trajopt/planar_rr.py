import matplotlib.pyplot as plt
import numpy as np
from gekko import GEKKO
from matplotlib import animation

thetaDotMin = np.deg2rad(-180)
thetaDotMax = np.deg2rad(180)
torqueMin = np.deg2rad(-50)
torqueMax = np.deg2rad(50)

m = GEKKO()

nt = 501
tm = np.linspace(0, 1, nt)
m.time = tm

# task space var
R = 0.15
t1I = 0
t2I = 0
px1 = m.Var(value=R * m.cos(t1I))
py1 = m.Var(value=R * m.sin(t1I))
px2 = m.Var(value=3 * R * m.cos(t1I))
py2 = m.Var(value=3 * R * m.sin(t1I))
px3 = m.Var(value=5 * R * m.cos(t1I))
py3 = m.Var(value=5 * R * m.sin(t1I))

px4 = m.Var(value=6 * R * m.cos(t1I) + R * m.cos(t1I + t2I))
py4 = m.Var(value=6 * R * m.sin(t1I) + R * m.sin(t1I + t2I))
px5 = m.Var(value=6 * R * m.cos(t1I) + 3 * R * m.cos(t1I + t2I))
py5 = m.Var(value=6 * R * m.sin(t1I) + 3 * R * m.sin(t1I + t2I))
px6 = m.Var(value=6 * R * m.cos(t1I) + 5 * R * m.cos(t1I + t2I))
py6 = m.Var(value=6 * R * m.sin(t1I) + 5 * R * m.sin(t1I + t2I))

# joint space var
theta1 = m.Var(value=0)
theta2 = m.Var(value=0)
thetaDot1 = m.Var(value=0, lb=thetaDotMin, ub=thetaDotMax)
thetaDot2 = m.Var(value=0, lb=thetaDotMin, ub=thetaDotMax)

# MV
uacc1 = m.MV(value=0, lb=torqueMin, ub=torqueMax)
uacc2 = m.MV(value=0, lb=torqueMin, ub=torqueMax)
uacc1.STATUS = 1
uacc2.STATUS = 1

# FV
tf = m.FV(value=20.0, lb=0.1, ub=100.0)
tf.STATUS = 1

# dynamic constraint
m.Equation(theta1.dt() == thetaDot1 * tf)
m.Equation(theta2.dt() == thetaDot2 * tf)
m.Equation(thetaDot1.dt() == uacc1 * tf)
m.Equation(thetaDot2.dt() == uacc2 * tf)

# forward kinematic equation constraint
m.Equation(px1 == R * m.cos(theta1))
m.Equation(py1 == R * m.sin(theta1))
m.Equation(px2 == 3 * R * m.cos(theta1))
m.Equation(py2 == 3 * R * m.sin(theta1))
m.Equation(px3 == 5 * R * m.cos(theta1))
m.Equation(py3 == 5 * R * m.sin(theta1))

m.Equation(px4 == 6 * R * m.cos(theta1) + R * m.cos(theta1 + theta2))
m.Equation(py4 == 6 * R * m.sin(theta1) + R * m.sin(theta1 + theta2))
m.Equation(px5 == 6 * R * m.cos(theta1) + 3 * R * m.cos(theta1 + theta2))
m.Equation(py5 == 6 * R * m.sin(theta1) + 3 * R * m.sin(theta1 + theta2))
m.Equation(px6 == 6 * R * m.cos(theta1) + 5 * R * m.cos(theta1 + theta2))
m.Equation(py6 == 6 * R * m.sin(theta1) + 5 * R * m.sin(theta1 + theta2))

m.fix(theta1, pos=len(m.time) - 1, val=np.deg2rad(90))
m.fix(theta2, pos=len(m.time) - 1, val=0)
m.fix(thetaDot1, pos=len(m.time) - 1, val=0)
m.fix(thetaDot2, pos=len(m.time) - 1, val=0)

considerCol = False

if considerCol:
    # collision constraint
    r = 0.2
    xc = 6 * R + 0.3
    yc = 6 * R + 0.3
    m.Equation(m.sqrt((px1 - xc) ** 2 + (py1 - yc) ** 2) >= r)
    m.Equation(m.sqrt((px2 - xc) ** 2 + (py2 - yc) ** 2) >= r)
    m.Equation(m.sqrt((px3 - xc) ** 2 + (py3 - yc) ** 2) >= r)
    m.Equation(m.sqrt((px4 - xc) ** 2 + (py4 - yc) ** 2) >= r)
    m.Equation(m.sqrt((px5 - xc) ** 2 + (py5 - yc) ** 2) >= r)
    m.Equation(m.sqrt((px6 - xc) ** 2 + (py6 - yc) ** 2) >= r)

m.Minimize(tf)

m.options.IMODE = 6
m.options.MAX_ITER = 1000
m.solve()

data = np.vstack((theta1.VALUE, theta2.VALUE, thetaDot1, thetaDot2, uacc1, uacc2))
# np.save('./datasave/joint_value/traj_opt.npy', data)
# data = np.load('./datasave/joint_value/traj_opt.npy')

fig, axs = plt.subplots(6, 1)
axs[0].plot(tm, data[0, :])
axs[1].plot(tm, data[1, :])
axs[2].plot(tm, data[2, :])
axs[3].plot(tm, data[3, :])
axs[4].plot(tm, data[4, :])
axs[5].plot(tm, data[5, :])
plt.show()


# Function to update the position of the circle in each frame
def update(frame):
    theta1 = data[0, frame]
    theta2 = data[1, frame]
    px1 = R * np.cos(theta1)
    py1 = R * np.sin(theta1)
    px2 = 3 * R * np.cos(theta1)
    py2 = 3 * R * np.sin(theta1)
    px3 = 5 * R * np.cos(theta1)
    py3 = 5 * R * np.sin(theta1)
    px4 = 6 * R * np.cos(theta1) + R * np.cos(theta1 + theta2)
    py4 = 6 * R * np.sin(theta1) + R * np.sin(theta1 + theta2)
    px5 = 6 * R * np.cos(theta1) + 3 * R * np.cos(theta1 + theta2)
    py5 = 6 * R * np.sin(theta1) + 3 * R * np.sin(theta1 + theta2)
    px6 = 6 * R * np.cos(theta1) + 5 * R * np.cos(theta1 + theta2)
    py6 = 6 * R * np.sin(theta1) + 5 * R * np.sin(theta1 + theta2)

    # Increment x and y coordinates to make the circle move
    circle1.center = (px1, py1)
    circle2.center = (px2, py2)
    circle3.center = (px3, py3)
    circle4.center = (px4, py4)
    circle5.center = (px5, py5)
    circle6.center = (px6, py6)
    return circle1, circle2, circle3, circle4, circle5, circle6


# Create a figure and axis
fig, ax = plt.subplots()
ax.set_aspect(1)
ax.set_xlim(-5, 5)
ax.set_ylim(-5, 5)

# Create a circle with initial position
circle1 = plt.Circle((0, 0), R, fc="b")
circle2 = plt.Circle((0, 0), R, fc="b")
circle3 = plt.Circle((0, 0), R, fc="b")
circle4 = plt.Circle((0, 0), R, fc="b")
circle5 = plt.Circle((0, 0), R, fc="b")
circle6 = plt.Circle((0, 0), R, fc="b")
ax.add_patch(circle1)
ax.add_patch(circle2)
ax.add_patch(circle3)
ax.add_patch(circle4)
ax.add_patch(circle5)
ax.add_patch(circle6)

if considerCol:
    circleCol = plt.Circle((xc, yc), r, fc="r")
    ax.add_patch(circleCol)

animation = animation.FuncAnimation(fig, update, frames=data.shape[1], interval=50, blit=True)
plt.show()
