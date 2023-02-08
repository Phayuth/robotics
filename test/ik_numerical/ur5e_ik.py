from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

theta = np.array([[0],
                  [0],
                  [0],
                  [0],
                  [0],
                  [0]])
r = ur5e.ur5e()
_, _, _, _, _, p = r.forward_kinematic(theta)

roll = 0
pitch = 0
yaw = 0
x_desired = np.array([[0.5],
                      [0.5],
                      [0.5],
                      [roll],
                      [pitch],
                      [yaw]])

x_current = np.array([[p[0, 0]],
                      [p[1, 0]],
                      [p[2, 0]],
                      [0],
                      [0],
                      [0]])

e = x_desired - x_current
max_iter = 100
i = 0

while np.linalg.norm(e) > 0.001 and i < max_iter:
    _, _, _, _, _, p = r.forward_kinematic(theta)
    x_current = np.array([[p[0, 0]],
                          [p[1, 0]],
                          [p[2, 0]],
                          [0],
                          [0],
                          [0]])
    e = x_desired - x_current
    Jac = r.jacobian(theta)
    del_theta = np.linalg.pinv(Jac) @ e
    theta = theta + del_theta
    print(theta)
    i += 1

print(i)


p01, p02, p03, p04, p05, p06 = r.forward_kinematic(theta)
print(p01)
print(p02)
print(p03)
print(p04)
print(p05)
print(p06)


fig = plt.figure(figsize=(4, 4))

ax = fig.add_subplot(111, projection='3d')

ax.scatter(0, 0, 0, label='origin')
ax.scatter(1, 0, 0, label='x')
ax.scatter(0, 1, 0, label='y')
ax.scatter(0, 0, 1, label='z')

ax.scatter(p01[0, 0], p01[1, 0], p01[2, 0], label='j1')
ax.scatter(p02[0, 0], p02[1, 0], p02[2, 0], label='j2')
ax.scatter(p03[0, 0], p03[1, 0], p03[2, 0], label='j3')
ax.scatter(p04[0, 0], p04[1, 0], p04[2, 0], label='j4')
ax.scatter(p05[0, 0], p05[1, 0], p05[2, 0], label='j5')
ax.scatter(p06[0, 0], p06[1, 0], p06[2, 0], label='j6')


ax.legend()
ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)
ax.set_zlim(-1, 1)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

plt.show()
