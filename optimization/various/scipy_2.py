import numpy as np
import scipy.optimize as sop
import matplotlib.pyplot as plt


def model(x):
    areal = 3.2
    breal = 2.5
    return areal * x + breal


noise = 1.2 * np.random.uniform(-1, 1, 100)
x = np.linspace(0, 10, 100)
y = model(x)

ynoise = y + noise


def objective_function(vopt):
    aopt = vopt[0]
    bopt = vopt[1]
    yhat = aopt * x + bopt
    err = yhat - ynoise
    cost = np.sum(np.square(np.abs(err)))
    return cost


initial_guess = [6.1, 5.8]
result = sop.minimize(objective_function, initial_guess, options={"disp": True})
print(f"> result: {result}")

vopt = result.x
yest = vopt[0] * x + vopt[1]

plt.plot(x, y, "r", label="model")
plt.plot(x, ynoise, "b.", label="model noise")
plt.plot(x, yest, "g", label="model estimated")
plt.legend()
plt.show()


arange = np.linspace(-30, 30, 100)
brange = np.linspace(-30, 30, 100)
A, B = np.meshgrid(arange, brange)



def cosst(a, b):
    yhat = a * x + b
    err = yhat - ynoise
    cost = np.sum(np.square(np.abs(err)))
    return cost

Z = []
f = []
for i in range(A.shape[0]):
    for j in range(B.shape[0]):
        f.append(cosst(A[i,j], B[i,j]))
    Z.append(f)
    f = []

Z = np.array(Z)
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
surf = ax.plot_surface(A, B, Z, linewidth=0, antialiased=False)
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.show()
