import numpy as np
import matplotlib.pyplot as plt
# https://medium.com/nerd-for-tech/local-path-planning-using-virtual-potential-field-in-python-ec0998f490af

# creating two evenly spaced array with ranging from 
# -10 to 10
size = 10
x = np.arange(-size,size,1)
y = np.arange(-size,size,1)

# Creating the meshgrid 
X, Y = np.meshgrid(x,y)

#Creating delx and dely array
delx = np.zeros_like(X)
dely = np.zeros_like(Y)
s = 2
r = 1
alpha = 50


delx = np.zeros_like(X)
dely = np.zeros_like(Y)
loc = np.array([6,10])


for i in range(len(x)):
    for j in range(len(y)):
        
        d= np.sqrt((loc[0]-X[i][j])**2 + (loc[1]-Y[i][j])**2)
        #print(f"{i} and {j}")
        theta = np.arctan2(loc[1]-Y[i][j], loc[0] - X[i][j])
        if d< r:
            delx[i][j] = 0
            dely[i][j] =0
        elif d>r+s:
            delx[i][j] = alpha* s *np.cos(theta)
            dely[i][j] = alpha * s *np.sin(theta)
        else:
            delx[i][j] = alpha * (d-r) *np.cos(theta)
            dely[i][j] = alpha * (d-r) *np.sin(theta)

fig, ax = plt.subplots(figsize = (10,10))
ax.quiver(X, Y, delx, dely)
ax.add_patch(plt.Circle((loc[0], loc[1]), r, color='b'))
ax.annotate("Goal", xy=(loc[0], loc[1]), fontsize=20, ha="center")
ax.set_title('Attractive field of the Goal')
plt.show() 