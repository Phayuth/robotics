import numpy as np
import matplotlib.pyplot as plt

def circle_fit(x,y):
    A = []
    for i in range(len(x)):
        row = [x[i],y[i],1]
        A.append(row)
    A = np.array(A)

    B = []
    for i in range(len(y)):
        row = [-x[i]**2 -y[i]**2]
        B.append(row)
    B = np.array(B)

    x = np.linalg.inv((np.transpose(A) @ A)) @ np.transpose(A) @ B # pseudo inverse

    a = -x[0]/2
    b = -x[1]/2
    e = x[2]

    r = np.sqrt(a**2+b**2-e)
    return a,b,r

# data
theta = np.linspace(0,2*np.pi,50)
r = 100

x = np.zeros(len(theta))
y = np.zeros(len(theta))

for i in range(len(theta)):
    x[i] = r*np.cos(theta[i]) + np.random.normal(0,1) + 10
    y[i] = r*np.sin(theta[i]) + np.random.normal(0,1) + 5

a,b,rd = circle_fit(x,y)
print(a,b,rd)

x_est = rd*np.cos(theta)
y_est = rd*np.sin(theta)

figure, axes = plt.subplots()
Drawing_colored_circle = plt.Circle((a,b),rd)
 
axes.set_aspect( 1 )
axes.add_artist( Drawing_colored_circle )
plt.title( 'Colored Circle' )
plt.scatter(x,y)
plt.show()