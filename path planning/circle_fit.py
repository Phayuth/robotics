import numpy as np
import matplotlib.pyplot as plt

# https://nu-msr.github.io/navigation_site/lectures/circle_fit.html#org05d39dc

def circle_fit(x,y):
    x_mean = np.sum(x)/len(x)
    y_mean = np.sum(y)/len(y)

    x_shift = x - x_mean
    y_shift = y - y_mean

    z = x_shift*x_shift + y_shift*y_shift

    z_mean = np.sum(z)/len(z)

    Z = []
    for i in range(len(z)):
        element = [x[i]*x[i] + y[i]*y[i] , x[i] , y[i] , 1]
        Z.append(element)
    Z = np.array(Z)

    M = (1/len(z))*(np.transpose(Z) @ Z)

    H = np.array([[8*z_mean,0,0,2],
                         [0,1,0,0],
                         [0,0,1,0],
                         [2,0,0,0]])

    H_inv = np.linalg.inv(H)
    U,sigma,VT = np.linalg.svd(Z)
    sigma_full = np.diag(sigma)
    V = np.transpose(VT)

    if sigma[3]<10**(-12):
        A = V[:,3]
        print("SMALL")

    elif sigma[3]>10**(-12):
        Y = V @ sigma_full @ VT
        Q = Y @ H_inv @ Y
        eigVal,eigVec = np.linalg.eig(Q)
        min_eig_val_index = np.argmin(eigVal)
        Astar = eigVec[:,min_eig_val_index]
        A = np.linalg.lstsq(Y,Astar,rcond=None)[0]
        print("BIG")

    a = -A[1]/(2*A[0])
    b = -A[2]/(2*A[0])
    Rterm = (A[1]*A[1] + A[2]*A[2] - 4*A[0]*A[3])/(4*A[0]*A[0])
    R = np.sqrt(Rterm)
    center = np.array([x_mean + a , y_mean + b])
    radius = R
    return center, radius

# data
theta = np.linspace(0,2*np.pi,50)
r = 3

x = np.zeros(len(theta))
y = np.zeros(len(theta))

for i in range(len(theta)):
    x[i] = r*np.cos(theta[i]) + np.random.normal(0,0.1)
    y[i] = r*np.sin(theta[i]) + np.random.normal(0,0.1)

center , radius = circle_fit(x,y)

print(center)
print(radius)
 
sg = []
for i in range(len(theta)):
    qqq = ((x[i]-center[0])**2 + (y[i]-center[1])**2 - radius**2)**2
    sg.append(qqq)

sgerr = np.sqrt(np.sum(sg)/len(theta))
print(sgerr)

# figure, axes = plt.subplots()
# Drawing_colored_circle = plt.Circle((center[0],center[1]),radius)
 
# axes.set_aspect( 1 )
# axes.add_artist( Drawing_colored_circle )
# plt.title( 'Colored Circle' )
plt.scatter(x,y)
plt.show()