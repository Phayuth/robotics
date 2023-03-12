import numpy as np
import matplotlib.pyplot as plt

x = np.arange(-10,10,1)
y = np.arange(-10,10,1)

X, Y = np.meshgrid(x,y)

delx = np.zeros_like(X)
dely = np.zeros_like(Y)

s = 7
r = 2
for i in range(len(x)):
  for j in range(len(y)):
    
    d= np.sqrt(X[i][j]**2 + Y[i][j]**2)
    #print(f"{i} and {j}")
    theta = np.arctan2(Y[i][j],X[i][j])
    if d< 2:
      delx[i][j] = np.sign(np.cos(theta)) +0
      dely[i][j] = np.sign(np.cos(theta)) +0
    elif d>r+s:
      delx[i][j] = 0 +(-50 * s *np.cos(theta))
      dely[i][j] = 0 + (-50 * s *np.sin(theta))
    else:
      delx[i][j] = 50 *(s+r-d)* np.cos(theta) + (-50 * (d-r) *np.cos(theta))
      dely[i][j] = 50 * (s+r-d)*  np.sin(theta) + (-50 * (d-r) *np.sin(theta))

fig, ax = plt.subplots(figsize = (10,10))
ax.quiver(X, Y, delx, dely)
ax.add_patch(plt.Circle((0, 0), 2, color='y'))

ax.annotate("Obstacle and\n Goal overlap", xy=(0, 0), fontsize=13, ha="center")
ax.set_title('Quiver plot if goal and obstacle both lie on Origin ')
plt.show() 