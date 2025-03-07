import numpy as np
import scipy.optimize as sop
import matplotlib.pyplot as plt


# simple path planning using optimization


def obj_func(xopt):
    q = np.array(xopt).reshape(-1, 2)
    err = np.diff(q, axis=0)
    n = np.linalg.norm(err, axis=1)
    nsq = np.square(n)
    cost = np.sum(nsq)
    return cost


def startconstr(x):
    qinit = np.array([0, 0])
    err = np.array([x[0], x[1]]) - qinit  # first 2 element norm must be equal to 0
    return err


def endconstr(x):
    qend = np.array([0, 5])
    err = np.array([x[-2], x[-1]]) - qend  # last 2 element norm
    return err


cons = [
    {"type": "eq", "fun": startconstr},
    {"type": "eq", "fun": endconstr},
]

initial_guess = [0, 0.5, 1, 1, 1, 2, 1, 3, 1, 4, 1, 5]  # 5 segments, 6 points, 12 elements
initial_guess = np.zeros(12)  # 5 segments, 6 points, 12 elements

bounds = [(-5, 5)] * len(initial_guess)
result = sop.minimize(obj_func, initial_guess, bounds=bounds, constraints=cons, method="SLSQP", options={"disp": True})
print(f"> result: {result}")

qopt = result.x
print(f"> qopt: {qopt}")

qopt = np.array(qopt).reshape(-1, 2)
x = qopt[:, 0]
y = qopt[:, 1]
plt.plot(x, y, "*-")
plt.axis("equal")
plt.show()
